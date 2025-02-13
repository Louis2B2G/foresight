import os
import math
import random
import requests
import json
import time
from typing import List, Optional, Literal, Dict, Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
import argparse

################################################################################
# --------------------------- DATA STRUCTURES --------------------------------- #
################################################################################

SpeakerType = Literal["User", "Correspondent"]

class Node:
    def __init__(
        self,
        from_speaker: SpeakerType,
        content: str,
        parent: Optional['Node'] = None,
        depth: int = 0
    ):
        self.from_speaker = from_speaker
        self.content = content
        self.parent = parent
        self.depth = depth
        self.children: List['Node'] = []
        self.is_terminal = False
        
        # MCTS specific attributes
        self.N = 0  # Number of visits
        self.Q = 0  # Total reward

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Reconstruct conversation history by traversing up to root.
        """
        history: List[Dict[str, str]] = []
        current_node = self
        
        while current_node is not None:
            if current_node.content:
                history.append({
                    'from': current_node.from_speaker,
                    'to': 'Correspondent' if current_node.from_speaker == 'User' else 'User',
                    'content': current_node.content
                })
            current_node = current_node.parent
        
        return history[::-1]  # Reverse to get chronological order

################################################################################
# --------------------------- UTILS ------------------------------------------- #
################################################################################

def load_conversation() -> List[Dict[str, str]]:
    """
    Load the initial conversation from a JSON file specified by `args.previous_conversation`.
    """
    try:
        with open(args.previous_conversation, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
            # Verify each message has the correct structure
            for msg in conversation:
                if not all(key in msg for key in ['from', 'to', 'content']):
                    raise ValueError(f"Invalid message format in {args.previous_conversation}")
            print("\nLoaded conversation from file:")
            for msg in conversation:
                print(f"{msg['from']}: {msg['content']}")
            return conversation
    except Exception as e:
        print(f"Error loading {args.previous_conversation}: {e}")
        return []

def parse_existing_conversation(conversation: List[Dict[str, str]]) -> Node:
    """
    Parse the loaded conversation into a root node containing the full history.
    """
    if not conversation:
        return Node(from_speaker="User", content="", parent=None, depth=0)
    
    conversation_text = "\n".join(f"{msg['from']}: {msg['content']}" for msg in conversation)
    root = Node(
        from_speaker="User",
        content=conversation_text,
        parent=None,
        depth=0
    )
    return root

def is_rate_limit_error(exception):
    return (
        isinstance(exception, requests.exceptions.RequestException)
        and getattr(exception.response, 'status_code', None) == 429
    )

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception(is_rate_limit_error)
)
def make_mistral_api_call(
    messages: List[dict],
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict] = None
) -> dict:
    """
    Makes an API call to the Mistral endpoint using an environment variable for auth.
    """
    api_token = os.environ.get("MISTRAL_API_TOKEN", "YOUR_TOKEN_HERE")
    if api_token == "YOUR_TOKEN_HERE":
        raise ValueError(
            "No valid MISTRAL_API_TOKEN found in environment. "
            "Set it via `export MISTRAL_API_TOKEN='your_token'` or pass it another way."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }
    
    payload = {
        "model": "mistral-large-latest",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    resp.raise_for_status()
    return resp.json()

def LLM_call(prompt: str, num_responses: int = 1) -> List[str]:
    """
    Calls the Mistral API with either a user prompt (multiple responses)
    or a single Correspondent's response.
    """
    try:
        if num_responses > 1:
            json_prompt = (
                f"Based on the following context, generate {num_responses} different response options "
                f"FROM THE USER'S PERSPECTIVE. Return them as a JSON object with a 'responses' field containing "
                f"an array of strings. These should be responses that the User would say.\n\nCONTEXT:\n{prompt}"
            )
        else:
            json_prompt = (
                f"Based on the following context, generate Correspondent's most likely response. "
                f"You should respond AS Correspondent. "
                f"Return it as a JSON object with a 'response' field containing a single string.\n\n"
                f"CONTEXT:\n{prompt}"
            )
        
        print("\n" + "="*80)
        print("SENDING PROMPT TO LLM:")
        print("-"*40)
        print(f"Number of responses requested: {num_responses}")
        print(f"Prompt type: {'User options' if num_responses > 1 else 'Correspondent response'}")
        print("\nFull prompt:")
        print("-"*40)
        print(json_prompt)
        print("="*80 + "\n")
        
        data = make_mistral_api_call(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
                {"role": "user", "content": json_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        response_text = data["choices"][0]["message"]["content"]
        try:
            resp_json = json.loads(response_text)
        except json.JSONDecodeError:
            resp_json = {}

        if num_responses > 1:
            responses = resp_json.get("responses", [])
        else:
            responses = [resp_json.get("response", "")]

        time.sleep(0.5)  # small delay for courtesy
        return responses[:num_responses]
    
    except Exception as e:
        print(f"Error calling Mistral API: {e}")
        return [f"Candidate response {i+1}" for i in range(num_responses)]

################################################################################
# --------------------------- MCTS LOGIC -------------------------------------- #
################################################################################

def get_conversation_messages(node: Node) -> List[dict]:
    return node.get_conversation_history()

def last_speaker_is_user(node: Node) -> bool:
    return node.depth % 2 == 0  # Even depths => User's turn

def expand_node(node: Node, num_candidates: int, max_depth: int, objective: str):
    """
    Expand a node by generating possible next moves.
    """
    print(f"\nExpanding node at depth {node.depth}")
    
    if node.depth >= max_depth:
        print("Reached max depth, marking as terminal.")
        node.is_terminal = True
        return

    conversation_so_far = node.get_conversation_history()
    next_speaker: SpeakerType = "Correspondent" if node.depth % 2 == 1 else "User"
    print(f"Next speaker: {next_speaker}")

    formatted_conversation = "\n".join(f"{msg['from']}: {msg['content']}" for msg in conversation_so_far)

    if next_speaker == "User":
        print(f"Generating {num_candidates} user responses.")
        prompt = (
            f"YOU ARE THE USER. Current conversation:\n{formatted_conversation}\n\n"
            f"Your objective is: {objective}\n\n"
            f"Generate {num_candidates} different responses from the USER'S perspective "
            f"that help achieve this objective. Make them natural, conversational, "
            f"and relevant to the context. Try to keep them distinct."
        )
        user_responses = LLM_call(prompt, num_candidates)
        print(f"Generated {len(user_responses)} responses.")
        
        for i, user_msg in enumerate(user_responses):
            user_node = Node(
                from_speaker="User",
                content=user_msg,
                parent=node,
                depth=node.depth + 1
            )
            node.children.append(user_node)
            print(f"Added user response {i+1}/{len(user_responses)}")
            
    else:  # Correspondent's turn
        print("Generating Correspondent response.")
        cat_prompt = (
            f"Current conversation:\n{formatted_conversation}\n\n"
            "You are Correspondent. Reply to the USER while maintaining the same style/tone."
        )
        cat_response = LLM_call(cat_prompt, 1)[0]
        print("Generated Correspondent response.")
        
        cat_node = Node(
            from_speaker="Correspondent",
            content=cat_response,
            parent=node,
            depth=node.depth + 1
        )
        node.children.append(cat_node)
        print("Added Correspondent response.")

    print(f"Node expansion complete. Generated {len(node.children)} children.")

def get_relative_score(scores: List[float]) -> List[float]:
    """
    Convert absolute scores to relative: best=1, worst=0, linearly in between.
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1])
    n = len(scores)
    rel = [0.0] * n
    for rank, (idx, val) in enumerate(indexed):
        rel[idx] = rank / (n - 1)
    return rel

def parse_llm_response(response_text: str, key: str, default: Any) -> Any:
    """
    Helper function to parse LLM JSON responses.
    """
    try:
        js = json.loads(response_text)
        return js.get(key, default)
    except:
        return default

def simulate(
    node: Node,
    num_candidates: int,
    max_depth: int,
    objective: str,
    use_relative_scoring: bool = False
) -> float:
    """
    Simulate a random path from 'node' to a leaf, then rate it.
    """
    print(f"\nStarting {'relative' if use_relative_scoring else 'absolute'} scoring simulation "
          f"at depth {node.depth}")
    
    if use_relative_scoring:
        print("Collecting multiple conversation states for ranking...")
        states = []
        for i in range(num_candidates):
            current = node
            while not current.is_terminal and current.depth < max_depth:
                if not current.children:
                    expand_node(current, num_candidates, max_depth, objective)
                if not current.children:
                    print(f"Warning: Node at depth {current.depth} has no children after expansion.")
                    break
                current = random.choice(current.children)
            states.append(current.content)
            print(f"Collected state {i+1}/{num_candidates}")

        eval_prompt = (
            f"You are an evaluator. For each conversation below, evaluate how well it achieves "
            f"the following objective and return a score between 0 and 1:\n"
            f"OBJECTIVE: {objective}\n\n"
        )
        for i, st in enumerate(states, 1):
            eval_prompt += f"\nCONVERSATION {i}:\n{st}\n"
        eval_prompt += "\nReturn a JSON object with scores as an array of numbers between 0 and 1."

        msgs = [
            {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
            {"role": "user", "content": eval_prompt}
        ]
        data = make_mistral_api_call(
            messages=msgs,
            temperature=0.8,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        scores = parse_llm_response(data["choices"][0]["message"]["content"], "scores", [])
        if len(scores) != len(states):
            return 0.0
        
        print("\nScores for each conversation:")
        print("-" * 40)
        for i, (state, score) in enumerate(zip(states, scores), 1):
            print(f"Conversation {i}:")
            print(f"Score: {score:.3f}")
            print("-" * 40)
        
        rel = get_relative_score(scores)
        return rel[0]

    else:
        print("Running single state evaluation...")
        current = node
        while not current.is_terminal and current.depth < max_depth:
            if not current.children:
                expand_node(current, num_candidates, max_depth, objective)
            if not current.children:
                print(f"Warning: Node at depth {current.depth} has no children after expansion.")
                break
            current = random.choice(current.children)
        
        eval_prompt = (
            f"Evaluate how well the conversation achieves this objective and return a score between 0 and 1:\n"
            f"OBJECTIVE: {objective}\n\n"
            f"CONVERSATION:\n{current.content}\n\n"
            "Return a JSON object with a single 'score' field containing a number between 0 and 1."
        )
        msgs = [
            {"role": "system", "content": "You are a helpful assistant that returns only valid JSON."},
            {"role": "user", "content": eval_prompt}
        ]
        data = make_mistral_api_call(
            messages=msgs,
            temperature=0.8,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        score = parse_llm_response(data["choices"][0]["message"]["content"], "score", 0.0)
        
        print("\nScore for conversation:")
        print("-" * 40)
        print(f"Score: {score:.3f}")
        print("-" * 40)
        return float(score)

def backpropagate(node: Node, reward: float, discount=1.0):
    """
    Update Q and N for each node up to the root.
    """
    current = node
    d = 0
    while current:
        current.N += 1
        current.Q += reward * (discount ** d)
        current = current.parent
        d += 1

def best_child(node: Node, c_param=1.0) -> Node:
    """
    Choose the best child using the UCT formula: Q/N + c * sqrt(ln(Nparent)/Nchild).
    """
    def uct_value(child: Node) -> float:
        if child.N == 0:
            return float('inf')
        exploit = child.Q / child.N
        explore = c_param * math.sqrt(math.log(node.N) / child.N)
        return exploit + explore
        
    return max(node.children, key=uct_value)

def mcts(
    root: Node,
    num_sims: int,
    num_candidates: int,
    max_depth: int,
    objective: str,
    c_param=1.0,
    use_relative_scoring=False
):
    """
    Monte Carlo Tree Search from 'root'.
    """
    print(f"\n{'='*80}")
    print(f"Starting MCTS with {num_sims} simulations")
    print(f"{'='*80}")

    for sim in range(num_sims):
        print(f"\nSimulation {sim+1}/{num_sims}")
        
        # 1) Selection
        node = root
        path = [node]
        while node.children:
            if node.is_terminal or node.depth == max_depth:
                break
            # If fewer children than num_candidates, we can expand
            if len(node.children) < num_candidates and not node.is_terminal:
                expand_node(node, num_candidates, max_depth, objective)
                break
            node = best_child(node, c_param)
            path.append(node)
        
        print(f"Selected path depth: {len(path)}")

        # 2) Expansion
        if node.depth < max_depth and not node.is_terminal:
            print(f"Expanding node at depth {node.depth}")
            expand_node(node, num_candidates, max_depth, objective)
            if node.children:
                node = random.choice(node.children)
                path.append(node)

        # 3) Simulation
        print(f"Running simulation from depth {node.depth}")
        reward = simulate(node, num_candidates, max_depth, objective, use_relative_scoring)
        print(f"Simulation reward: {reward:.3f}")

        # 4) Backprop
        print("Backpropagating results...")
        backpropagate(node, reward)

        # Print current best path from the root
        if not root.children:
            print("No children at the root yet.")
        else:
            best_at_root = max(
                root.children,
                key=lambda ch: ch.Q / ch.N if ch.N > 0 else float('-inf')
            )
            if best_at_root.N > 0:
                val = best_at_root.Q / best_at_root.N
                print(f"Current best response (Q/N): {val:.3f} ({best_at_root.N} visits)")
            else:
                print("No visits for the root's children yet.")

    print(f"\n{'='*80}")
    print("MCTS completed")
    print(f"{'='*80}")

    if not root.children:
        return "No response found (no children in root)", root
    
    # Best child from root
    best_node = max(
        root.children,
        key=lambda ch: ch.Q / ch.N if ch.N>0 else float('-inf')
    )

    final_action = f"{best_node.from_speaker}: {best_node.content}"
    return final_action, best_node

################################################################################
# --------------------------- MAIN + VISUAL ----------------------------------- #
################################################################################

def save_tree_diagram(root: Node, filename: str = "mcts_tree") -> None:
    """
    Build a diagram of the conversation tree using graphviz.
    """
    try:
        import graphviz
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        return

    dot = graphviz.Digraph(comment="MCTS Tree")
    dot.attr(rankdir='TB')

    def add_node_edges(n: Node, parent_id=None):
        node_id = str(id(n))
        avg_value = n.Q / n.N if n.N > 0 else 0.0

        lines = [
            f"Speaker: {n.from_speaker}",
            f"Depth: {n.depth}",
            f"Visits: {n.N}",
            f"Value: {avg_value:.2f}",
            ""
        ]
        if n.content:
            lines.append(f"Content: {n.content}")

        label = "\n".join(lines)
        
        if parent_id is None:
            dot.node(node_id, label, shape="box", style="filled", fillcolor="lightgray")
        else:
            fillcolor = "lightblue" if n.from_speaker == "User" else "lightpink"
            dot.node(node_id, label, shape="box", style="filled", fillcolor=fillcolor)
            dot.edge(parent_id, node_id)
        
        for c in n.children:
            add_node_edges(c, node_id)

    add_node_edges(root, None)
    try:
        dot.render(filename, view=True, format='pdf', cleanup=True)
        print(f"Tree diagram saved as {filename}.pdf")
    except Exception as e:
        print(f"Error saving diagram: {e}")

def main(
    objective: str,
    use_relative_scoring: bool = False,
    max_depth: int = 3,
    num_simulations: int = 50,
    number_options: int = 2
):
    print(f"\n{'='*80}")
    print("STARTING CONVERSATION OPTIMIZATION")
    print(f"{'='*80}")
    print(f"\nObjective: {objective}")
    print(f"Scoring method: {'Relative' if use_relative_scoring else 'Absolute'}")
    print(f"Maximum additional depth: {max_depth}")
    print(f"Number of simulations: {num_simulations}")
    print(f"Number of user response options: {number_options}")

    # 1) Load conversation from JSON
    existing_conversation = load_conversation()
    
    print("\nInitial conversation state:")
    print("="*80)
    print(existing_conversation)
    print("="*80, "\n")

    # 2) Parse it into a chain of nodes
    root = parse_existing_conversation(existing_conversation)
    
    total_max_depth = root.depth + max_depth
    print(f"Starting from depth {root.depth}, total maximum depth will be: {total_max_depth}")

    # 3) Run MCTS
    best_response, best_node = mcts(
        root=root,
        num_sims=num_simulations,
        num_candidates=number_options,
        max_depth=total_max_depth,
        objective=objective,
        c_param=1.0,
        use_relative_scoring=use_relative_scoring
    )

    print("\nFINAL RESULTS")
    print("="*80)
    print("Best response selected:")
    print("="*40)
    print(best_response)
    print("="*40)

    print("\nFinal conversation state:")
    print("="*40)
    print(best_node.content if best_node else "No conversation.")
    print("="*40)

    print("\nGenerating tree visualization...")
    save_tree_diagram(root, "mcts_conversation_tree")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run conversation optimization using MCTS')
    
    parser.add_argument(
        '--objective',
        type=str,
        default="You want to make the conversation go as well as possible.",
        help='The conversation objective to optimize for'
    )
    
    parser.add_argument(
        '--previous-conversation',
        type=str,
        default='conversation.json',
        help='Path to the JSON file containing the previous conversation'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        help='Maximum additional depth of the conversation tree'
    )
    
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=30,
        help='Number of MCTS simulations to run'
    )
    
    parser.add_argument(
        '--num-options',
        type=int,
        default=5,
        help='Number of response options to consider at each step'
    )
    
    parser.add_argument(
        '--relative-scoring',
        action='store_true',
        help='Use relative scoring instead of absolute scoring'
    )
    
    global args
    args = parser.parse_args()
    
    main(
        objective=args.objective,
        use_relative_scoring=args.relative_scoring,
        max_depth=args.max_depth,
        num_simulations=args.num_simulations,
        number_options=args.num_options
    )

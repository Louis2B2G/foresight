# Conversation Optimizer using Monte Carlo Tree Search (MCTS)

Implementation of an AI-powered conversation optimizer that uses Monte Carlo Tree Search (MCTS) to help users make better conversational decisions. It analyzes past conversations and suggests optimal responses based on specified objectives.

The system generates detailed conversation trees as PDF visualizations (`mcts_conversation_tree.pdf`) that show:
- Complete conversation flow with all possible branches
- Success probability for each response
- Node statistics (visits, value estimates)

## Overview

The Conversation Optimizer helps you:
- Analyze existing chat conversations
- Generate and evaluate multiple possible responses
- Find optimal conversation paths using MCTS
- Visualize decision trees of possible conversation flows

## How It Works

### 1. Core Components

- **MCTS Implementation**: Uses Monte Carlo Tree Search to explore possible conversation paths and find optimal responses
- **LLM Integration**: Leverages Mistral's Large Language Model API for:
  - Generating candidate responses
  - Evaluating conversation quality
  - Scoring different conversation paths
- **Conversation Tree**: Builds and maintains a tree structure of possible conversation flows
- **Visualization**: Generates visual diagrams of the conversation decision tree using Graphviz

### 2. The MCTS Process

1. **Selection**: Traverses the existing tree based on UCT (Upper Confidence Bound for Trees) values
2. **Expansion**: Generates new possible responses using the LLM
3. **Simulation**: Plays out random conversations to evaluate outcomes
4. **Backpropagation**: Updates node statistics based on simulation results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conversation-optimizer.git
   cd conversation-optimizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Mistral API key:
   ```bash
   export MISTRAL_API_TOKEN=your_api_key
   ```

## Usage

### Basic Usage

```bash
   python main.py --objective "Make plans to meet up" --previous-conversation conversation.json
```

### Advanced Options
```bash
   python main.py \
--objective "Make plans to meet up" \
--previous-conversation conversation.json \
--max-depth 4 \
--num-simulations 30 \
--num-options 5 \
--relative-scoring
```

### Command Line Arguments

- `--objective`: The goal you want to achieve in the conversation
- `--previous-conversation`: Path to JSON file containing conversation history
- `--max-depth`: Maximum additional depth of the conversation tree
- `--num-simulations`: Number of MCTS simulations to run
- `--num-options`: Number of response options to consider at each step
- `--relative-scoring`: Use relative scoring instead of absolute scoring

## Input Format

The conversation history should be in JSON format:

```json
[
{
"from": "User",
"to": "Correspondent",
"content": "Message content"
},
{
"from": "Correspondent",
"to": "User",
"content": "Response content"
}
]
```

## Output

1. **Console Output**: 
   - Detailed progress of the MCTS process
   - Best suggested responses
   - Evaluation scores

2. **Visual Output**:
   - Generates a PDF diagram (`mcts_conversation_tree.pdf`) showing:
     - Conversation tree structure
     - Node statistics (visits, values)
     - Different conversation paths

## Use Cases

- **Dating Apps**: Optimize responses in dating app conversations
- **Professional Networking**: Improve LinkedIn or email communications
- **Customer Service**: Train customer service representatives
- **Social Media**: Plan content and responses for engagement

## Technical Details

### Scoring Methods

1. **Absolute Scoring**: Each conversation path is scored independently
2. **Relative Scoring**: Conversations are ranked against each other

### Tree Node Structure

Each node contains:
- Speaker information
- Message content
- Visit count (N)
- Total reward (Q)
- Children nodes
- Parent reference

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is meant to assist in conversation optimization but should not replace human judgment. Always review and adjust suggested responses based on context and appropriateness.
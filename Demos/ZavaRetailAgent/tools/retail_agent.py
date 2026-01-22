"""
Simplified Microsoft Foundry Agent with MCP Server Integration

This is a working example that creates an agent and demonstrates the API usage.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import RunHandler, ToolApproval
from colorama import init, Fore, Back, Style

# Initialize colorama for Windows color support
init(autoreset=True)

# Load environment variables
load_dotenv()


# System prompt from policy.md
SYSTEM_PROMPT = """# Retail agent policy

As a retail agent, you can help users:

- **cancel or modify pending orders**
- **return or exchange delivered orders**
- **modify their default user address**
- **provide information about their own profile, orders, and related products**

At the beginning of the conversation, you have to authenticate the user identity by locating their user id via email, or via name + zip code. This has to be done even when the user already provides the user id.

Once the user has been authenticated, you can provide the user with information about order, product, profile information, e.g. help the user look up order id.

You can only help one user per conversation (but you can handle multiple requests from the same user), and must deny any requests for tasks related to any other user.

Before taking any action that updates the database (cancel, modify, return, exchange), you must list the action details and obtain explicit user confirmation (yes) to proceed.

You should not make up any information or knowledge or procedures not provided by the user or the tools, or give subjective recommendations or comments.

You should at most make one tool call at a time, and if you take a tool call, you should not respond to the user at the same time. If you respond to the user, you should not make a tool call at the same time.

You should deny user requests that are against this policy.

You should transfer the user to a human agent if and only if the request cannot be handled within the scope of your actions. To transfer, first make a tool call to transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.' to the user.

## Domain basic

- All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30 AM EST.

### User

Each user has a profile containing:

- unique user id
- email
- default address
- payment methods.

There are three types of payment methods: **gift card**, **paypal account**, **credit card**.

### Product

Our retail store has 50 types of products.

For each **type of product**, there are **variant items** of different **options**.

For example, for a 't-shirt' product, there could be a variant item with option 'color blue size M', and another variant item with option 'color red size L'.

Each product has the following attributes:

- unique product id
- name
- list of variants

Each variant item has the following attributes:

- unique item id
- information about the value of the product options for this item.
- availability
- price

Note: Product ID and Item ID have no relations and should not be confused!

### Order

Each order has the following attributes:

- unique order id
- user id
- address
- items ordered
- status
- fullfilments info (tracking id and item ids)
- payment history

The status of an order can be: **pending**, **processed**, **delivered**, or **cancelled**.

Orders can have other optional attributes based on the actions that have been taken (cancellation reason, which items have been exchanged, what was the exchane price difference etc)

## Generic action rules

Generally, you can only take action on pending or delivered orders.

Exchange or modify order tools can only be called once per order. Be sure that all items to be changed are collected into a list before making the tool call!!!

## Cancel pending order

An order can only be cancelled if its status is 'pending', and you should check its status before taking the action.

The user needs to confirm the order id and the reason (either 'no longer needed' or 'ordered by mistake') for cancellation. Other reasons are not acceptable.

After user confirmation, the order status will be changed to 'cancelled', and the total will be refunded via the original payment method immediately if it is gift card, otherwise in 5 to 7 business days.

## Modify pending order

An order can only be modified if its status is 'pending', and you should check its status before taking the action.

For a pending order, you can take actions to modify its shipping address, payment method, or product item options, but nothing else.

### Modify payment

The user can only choose a single payment method different from the original payment method.

If the user wants the modify the payment method to gift card, it must have enough balance to cover the total amount.

After user confirmation, the order status will be kept as 'pending'. The original payment method will be refunded immediately if it is a gift card, otherwise it will be refunded within 5 to 7 business days.

### Modify items

This action can only be called once, and will change the order status to 'pending (items modifed)'. The agent will not be able to modify or cancel the order anymore. So you must confirm all the details are correct and be cautious before taking this action. In particular, remember to remind the customer to confirm they have provided all the items they want to modify.

For a pending order, each item can be modified to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

## Return delivered order

An order can only be returned if its status is 'delivered', and you should check its status before taking the action.

The user needs to confirm the order id and the list of items to be returned.

The user needs to provide a payment method to receive the refund.

The refund must either go to the original payment method, or an existing gift card.

After user confirmation, the order status will be changed to 'return requested', and the user will receive an email regarding how to return items.

## Exchange delivered order

An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action. In particular, remember to remind the customer to confirm they have provided all items to be exchanged.

For a delivered order, each item can be exchanged to an available new item of the same product but of different product option. There cannot be any change of product types, e.g. modify shirt to shoe.

The user must provide a payment method to pay or receive refund of the price difference. If the user provides a gift card, it must have enough balance to cover the price difference.

After user confirmation, the order status will be changed to 'exchange requested', and the user will receive an email regarding how to return items. There is no need to place a new order.
"""


class AutoApproveRunHandler(RunHandler):
    """Custom RunHandler that auto-approves all MCP tool calls and shows tool execution."""
    
    def __init__(self):
        super().__init__()
        self.tool_calls = []
        self.tool_outputs = {}  # Store tool outputs by tool_call_id
    
    def submit_mcp_tool_approval(self, *, run, tool_call, **kwargs):
        """Auto-approve all MCP tool calls and display the request."""
        try:
            # Parse arguments
            args = {}
            if tool_call.arguments:
                try:
                    args = json.loads(tool_call.arguments)
                except:
                    args = {"raw": tool_call.arguments}
            
            # Store tool call info for later display with response
            self.tool_calls.append({
                'name': tool_call.name,
                'arguments': args,
                'id': tool_call.id
            })
            
            # Create and return approval
            approval = ToolApproval()
            approval['approve'] = True
            approval['tool_call_id'] = tool_call.id
            
            # Store for later retrieval and display with output
            self.tool_outputs[tool_call.id] = {
                'name': tool_call.name,
                'arguments': args
            }
            
            return approval
        except Exception as e:
            print(f"{Fore.RED}Error in submit_mcp_tool_approval: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return None


class ConversationLogger:
    """Logger to save conversation history to disk."""
    
    def __init__(self, base_dir="../conversations"):
        self.base_dir = base_dir
        self.conversation = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "model": None,
                "temperature": None,
                "top_p": None
            },
            "messages": []
        }
        self.filepath = None
        self.daily_jsonl_filepath = None
        
    def initialize(self, model, temperature, top_p, seed=None):
        """Initialize conversation with metadata."""
        # Create conversations directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.base_dir, f"conversation_{timestamp}.json")
        
        # Daily JSONL file (one file per day, appended across conversations)
        daily_date = datetime.now().strftime("%Y%m%d")
        self.daily_jsonl_filepath = os.path.join(self.base_dir, f"conversations_{daily_date}.jsonl")
        
        self.conversation["metadata"]["model"] = model
        self.conversation["metadata"]["temperature"] = temperature
        self.conversation["metadata"]["top_p"] = top_p
        if seed is not None:
            self.conversation["metadata"]["seed"] = seed
            self.conversation["metadata"]["deterministic_mode"] = True
    
    def add_user_message(self, content):
        """Add a user message to the conversation."""
        message = {
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation["messages"].append(message)
    
    def add_tool_call(self, tool_name, arguments, output):
        """Add a tool call with its output."""
        message = {
            "role": "tool",
            "tool_name": tool_name,
            "arguments": arguments,
            "output": output,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation["messages"].append(message)
    
    def add_assistant_message(self, content):
        """Add an assistant message to the conversation."""
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation["messages"].append(message)
    
    def save(self):
        """Save the conversation to disk (JSON and JSONL formats)."""
        if self.filepath:
            self.conversation["metadata"]["end_time"] = datetime.now().isoformat()
            
            # Save detailed JSON file with metadata
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversation, f, indent=2, ensure_ascii=False)
            
            # Append to daily JSONL file (entire conversation as one line)
            if self.daily_jsonl_filepath:
                try:
                    conversation_line = {"messages": self.conversation["messages"]}
                    with open(self.daily_jsonl_filepath, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(conversation_line, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Warning: Could not write to daily JSONL file: {e}")
            
            return self.filepath, self.daily_jsonl_filepath
        return None, None


def main(model_name=None, seed=None):
    """Main function to create and test the retail agent.
    
    Args:
        model_name: Optional model deployment name to use. If provided, overrides AZURE_OPENAI_DEPLOYMENT_NAME from .env
        seed: Optional seed for deterministic outputs. If provided, enables deterministic mode with temperature=0
    """
    print(f"{Fore.GREEN}{Style.BRIGHT}{'=' * 70}")
    print(f"🚀 RETAIL AGENT - Microsoft Foundry with MCP Integration")
    print(f"{'=' * 70}{Style.RESET_ALL}\n")
    
    # Initialize conversation logger
    logger = ConversationLogger()
    
    # Get configuration
    connection_string = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
    deployment_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    
    # Get model settings with defaults
    # If seed is provided, use temperature=0 for deterministic outputs
    if seed is not None:
        temperature = 0.0
        top_p = 1.0
    else:
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        top_p = float(os.getenv("TOP_P", "0.95"))
    
    if not connection_string:
        print(f"{Fore.RED}❌ Error: AZURE_AI_PROJECT_CONNECTION_STRING is required in .env file{Style.RESET_ALL}")
        print("\nPlease add your Microsoft Foundry project connection string to the .env file:")
        print("AZURE_AI_PROJECT_CONNECTION_STRING=your-connection-string-here")
        return
    
    if not mcp_server_url:
        print(f"{Fore.RED}❌ Error: MCP_SERVER_URL is required in .env file{Style.RESET_ALL}")
        print("\nPlease add your MCP server URL to the .env file:")
        print("MCP_SERVER_URL=http://retail-mcp-server.eastus.azurecontainer.io:8000/mcp")
        return
    
    try:
        # Initialize Azure AI Project Client
        print(f"{Fore.BLUE}📡 Connecting to Microsoft Foundry project...{Style.RESET_ALL}")
        credential = DefaultAzureCredential()
        
        # Use the full connection string as endpoint
        project_client = AIProjectClient(
            endpoint=connection_string,
            credential=credential
        )
        
        print(f"{Fore.GREEN}✅ Connected to Microsoft Foundry project{Style.RESET_ALL}")
        
        # Create agent with MCP tool
        print(f"\n{Fore.BLUE}🤖 Creating agent with deployment: {Style.BRIGHT}{deployment_name}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}   Temperature: {temperature}, Top P: {top_p}{Style.RESET_ALL}")
        if seed is not None:
            print(f"{Fore.BLUE}   Seed: {seed} (deterministic mode enabled via temperature=0){Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Note: Azure AI Agents SDK doesn't support seed parameter directly{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}   Using temperature=0 and top_p=1.0 for maximum determinism{Style.RESET_ALL}")
        
        # Create agent (note: seed is NOT passed here, only to individual runs)
        agent = project_client.agents.create_agent(
            model=deployment_name,
            name="Retail Assistant",
            instructions=SYSTEM_PROMPT,
            tools=[{
                "type": "mcp",
                "server_label": "retail_mcp_server",
                "server_url": mcp_server_url
            }],
            temperature=temperature,
            top_p=top_p
        )
        
        print(f"✅ Created agent: {agent.id}")
        
        # Initialize logger with metadata
        logger.initialize(deployment_name, temperature, top_p, seed)
        
        # Create conversation thread
        print(f"💬 Creating conversation thread...")
        thread = project_client.agents.threads.create()
        print(f"✅ Created thread: {thread.id}")
        
        # Interactive chat loop
        print(f"\n{Fore.GREEN}{Style.BRIGHT}{'=' * 70}")
        print("💬 RETAIL ASSISTANT CHAT")
        print(f"{'=' * 70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Type 'quit' or 'exit' to end the conversation{Style.RESET_ALL}\n")
        
        while True:
            try:
                user_input = input(f"{Fore.WHITE}{Back.BLUE} YOU {Style.RESET_ALL} ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print(f"\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                    break
                
                # Log user message
                logger.add_user_message(user_input)
                
                # Add user message
                project_client.agents.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=user_input
                )
                
                # Run the agent with auto-approve handler
                # Note: The Azure AI Agents SDK doesn't support seed parameter directly
                # Determinism is achieved through temperature=0 and top_p=1.0 when seed is provided
                run_handler = AutoApproveRunHandler()
                run = project_client.agents.runs.create_and_process(
                    thread_id=thread.id,
                    agent_id=agent.id,
                    run_handler=run_handler
                )
                
                # Check run status
                if run.status == "failed":
                    print(f"\n{Fore.RED}❌ Run failed: {run.last_error}{Style.RESET_ALL}")
                    continue
                
                # Display tool outputs from RunHandler
                if run_handler.tool_outputs:
                    try:
                        run_steps = project_client.agents.run_steps.list(
                            thread_id=thread.id,
                            run_id=run.id
                        )
                        
                        # Convert to list and reverse to get chronological order (oldest first)
                        steps_list = list(run_steps)
                        steps_list.reverse()
                        
                        for step in steps_list:
                            if hasattr(step, 'step_details') and step.step_details:
                                if hasattr(step.step_details, 'tool_calls') and step.step_details.tool_calls:
                                    for tool_call in step.step_details.tool_calls:
                                        # Check if this is an MCP tool call and we tracked it
                                        if hasattr(tool_call, 'output') and tool_call.id in run_handler.tool_outputs:
                                            tool_info = run_handler.tool_outputs[tool_call.id]
                                            
                                            # Display tool call header
                                            print(f"\n{Fore.CYAN}{'━' * 70}")
                                            print(f"{Fore.CYAN}� MCP TOOL CALL: {Style.BRIGHT}{tool_info['name']}{Style.RESET_ALL}")
                                            print(f"{Fore.CYAN}{'━' * 70}{Style.RESET_ALL}")
                                            
                                            # Display request
                                            print(f"{Fore.YELLOW}� REQUEST (JSON):{Style.RESET_ALL}")
                                            if tool_info['arguments']:
                                                request_json = json.dumps(tool_info['arguments'], indent=2)
                                                for line in request_json.split('\n'):
                                                    print(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
                                            else:
                                                print(f"{Fore.YELLOW}  (no parameters){Style.RESET_ALL}")
                                            
                                            # Display response
                                            print(f"\n{Fore.GREEN}📥 RESPONSE (JSON):{Style.RESET_ALL}")
                                            try:
                                                # Parse and format the output
                                                response_data = json.loads(tool_call.output) if isinstance(tool_call.output, str) else tool_call.output
                                                formatted_json = json.dumps(response_data, indent=2)
                                                for line in formatted_json.split('\n'):
                                                    print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
                                            except:
                                                # If parsing fails, just display the raw output
                                                response_data = tool_call.output
                                                print(f"{Fore.GREEN}{tool_call.output}{Style.RESET_ALL}")
                                            
                                            print(f"{Fore.CYAN}{'━' * 70}{Style.RESET_ALL}\n")
                                            
                                            # Log tool call
                                            logger.add_tool_call(
                                                tool_info['name'],
                                                tool_info['arguments'],
                                                response_data
                                            )
                    except Exception as e:
                        # Debug output if needed
                        print(f"{Fore.RED}DEBUG: Could not retrieve tool outputs: {e}{Style.RESET_ALL}")
                        import traceback
                        traceback.print_exc()
                
                # Get messages
                
                # Get messages
                messages = project_client.agents.messages.list(thread_id=thread.id)
                
                # Print latest assistant message
                print(f"{Fore.MAGENTA}{'━' * 70}")
                print(f"{Fore.MAGENTA}{Back.MAGENTA}{Fore.WHITE} ASSISTANT {Style.RESET_ALL}{Fore.MAGENTA} RESPONSE")
                print(f"{'━' * 70}{Style.RESET_ALL}")
                assistant_message = None
                for message in messages:
                    if message.role == "assistant":
                        for content in message.content:
                            if hasattr(content, 'text'):
                                assistant_message = content.text.value
                                print(f"{Fore.MAGENTA}{assistant_message}{Style.RESET_ALL}")
                                break
                        break
                
                # Log assistant message
                if assistant_message:
                    logger.add_assistant_message(assistant_message)
                
                print()
                
            except KeyboardInterrupt:
                print(f"\n\n{Fore.YELLOW}👋 Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
                import traceback
                traceback.print_exc()
        
        # Save conversation log
        saved_json, saved_jsonl = logger.save()
        if saved_json:
            print(f"\n{Fore.BLUE}💾 Conversation saved to:{Style.RESET_ALL}")
            print(f"   {Fore.CYAN}JSON: {Style.BRIGHT}{saved_json}{Style.RESET_ALL}")
            print(f"   {Fore.CYAN}JSONL: {Style.BRIGHT}{saved_jsonl}{Style.RESET_ALL}")
        
        # Cleanup
        print(f"\n{Fore.BLUE}🗑️  Cleaning up resources...{Style.RESET_ALL}")
        project_client.agents.delete_agent(agent.id)
        print(f"{Fore.GREEN}✅ Deleted agent: {agent.id}{Style.RESET_ALL}")
        project_client.agents.threads.delete(thread.id)
        print(f"{Fore.GREEN}✅ Deleted thread: {thread.id}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Retail Agent with MCP Integration')
    parser.add_argument('--model', type=str, help='Model deployment name (overrides .env)')
    parser.add_argument('--seed', type=int, help='Seed for deterministic outputs (enables temperature=0)')
    
    args = parser.parse_args()
    
    if args.model:
        print(f"{Fore.CYAN}Using model from command line: {Style.BRIGHT}{args.model}{Style.RESET_ALL}\n")
    if args.seed is not None:
        print(f"{Fore.CYAN}Using seed for deterministic mode: {Style.BRIGHT}{args.seed}{Style.RESET_ALL}\n")
    
    main(model_name=args.model, seed=args.seed)

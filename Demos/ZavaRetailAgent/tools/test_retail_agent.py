"""
Retail Agent Test Script

Tests the Microsoft Foundry retail agent with various scenarios.
Can be imported and used from Jupyter notebooks or run standalone.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import RunHandler, ToolApproval
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load environment variables
load_dotenv()


class AutoApproveRunHandler(RunHandler):
    """Auto-approve all MCP tool calls."""
    
    def __init__(self):
        super().__init__()
        self.tool_calls = []
        self.tool_outputs = {}
    
    def submit_mcp_tool_approval(self, *, run, tool_call, **kwargs):
        """Auto-approve MCP tool calls with correct signature."""
        try:
            # Parse arguments
            args = {}
            if tool_call.arguments:
                try:
                    args = json.loads(tool_call.arguments)
                except:
                    args = {"raw": tool_call.arguments}
            
            # Store tool call info
            self.tool_calls.append({
                'name': tool_call.name,
                'arguments': args,
                'id': tool_call.id
            })
            
            # Create and return approval
            approval = ToolApproval()
            approval['approve'] = True
            approval['tool_call_id'] = tool_call.id
            
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


class RetailAgentTester:
    """Test the retail agent with predefined scenarios."""
    
    def __init__(self, connection_string=None, model_name=None, mcp_server_url=None):
        """Initialize the tester with Azure configuration."""
        self.connection_string = connection_string or os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
        self.model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.mcp_server_url = mcp_server_url or os.getenv("MCP_SERVER_URL")
        
        # Ensure MCP URL ends with /mcp
        if self.mcp_server_url and not self.mcp_server_url.endswith('/mcp'):
            print(f"{Fore.YELLOW}⚠ MCP_SERVER_URL should end with '/mcp'. Adjusting...{Style.RESET_ALL}")
            self.mcp_server_url = self.mcp_server_url.rstrip('/') + '/mcp'
            print(f"{Fore.CYAN}  Using: {self.mcp_server_url}{Style.RESET_ALL}")
        
        self.results = {}
        self.project_client = None
        self.agent = None
        self.thread = None
        
        # System prompt (simplified version)
        self.system_prompt = """You are a helpful retail customer service agent. 
Help customers with their orders, returns, and account information."""
    
    def test_connection(self) -> bool:
        """Test Microsoft Foundry connection."""
        print(f"{Fore.CYAN}Testing Microsoft Foundry connection...{Style.RESET_ALL}")
        
        if not self.connection_string:
            print(f"{Fore.RED}✗ Missing AZURE_AI_PROJECT_CONNECTION_STRING{Style.RESET_ALL}")
            return False
        
        if not self.mcp_server_url:
            print(f"{Fore.RED}✗ Missing MCP_SERVER_URL{Style.RESET_ALL}")
            return False
        
        try:
            credential = DefaultAzureCredential()
            self.project_client = AIProjectClient(
                endpoint=self.connection_string,
                credential=credential
            )
            print(f"{Fore.GREEN}✓ Connected to Microsoft Foundry{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}✗ Connection failed: {e}{Style.RESET_ALL}")
            return False
    
    def test_agent_creation(self) -> bool:
        """Test agent creation with MCP tools."""
        print(f"\n{Fore.CYAN}Testing agent creation...{Style.RESET_ALL}")
        
        if not self.project_client:
            print(f"{Fore.RED}✗ Project client not initialized{Style.RESET_ALL}")
            return False
        
        try:
            self.agent = self.project_client.agents.create_agent(
                model=self.model_name,
                name="Test Retail Agent",
                instructions=self.system_prompt,
                tools=[{
                    "type": "mcp",
                    "server_label": "retail_mcp_server",
                    "server_url": self.mcp_server_url
                }],
                temperature=0.0,
                top_p=1.0
            )
            print(f"{Fore.GREEN}✓ Agent created: {self.agent.id}{Style.RESET_ALL}")
            print(f"  Model: {self.model_name}")
            return True
        except Exception as e:
            print(f"{Fore.RED}✗ Agent creation failed: {e}{Style.RESET_ALL}")
            return False
    
    def test_thread_creation(self) -> bool:
        """Test conversation thread creation."""
        print(f"\n{Fore.CYAN}Testing thread creation...{Style.RESET_ALL}")
        
        if not self.project_client:
            print(f"{Fore.RED}✗ Project client not initialized{Style.RESET_ALL}")
            return False
        
        try:
            self.thread = self.project_client.agents.threads.create()
            print(f"{Fore.GREEN}✓ Thread created: {self.thread.id}{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.RED}✗ Thread creation failed: {e}{Style.RESET_ALL}")
            return False
    
    def test_simple_query(self) -> bool:
        """Test a simple user query."""
        print(f"\n{Fore.CYAN}Testing simple query...{Style.RESET_ALL}")
        
        if not self.project_client or not self.agent or not self.thread:
            print(f"{Fore.RED}✗ Prerequisites not met{Style.RESET_ALL}")
            return False
        
        try:
            query = "What products do you have?"
            print(f"  Query: '{query}'")
            
            # Add message
            self.project_client.agents.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=query
            )
            
            # Run agent with auto-approve handler
            run_handler = AutoApproveRunHandler()
            run = self.project_client.agents.runs.create_and_process(
                thread_id=self.thread.id,
                agent_id=self.agent.id,
                run_handler=run_handler
            )
            
            if run.status == "completed":
                # Get response
                messages = self.project_client.agents.messages.list(thread_id=self.thread.id)
                for message in messages:
                    if message.role == "assistant":
                        for content in message.content:
                            if hasattr(content, 'text'):
                                response = content.text.value
                                print(f"{Fore.GREEN}✓ Got response{Style.RESET_ALL}")
                                print(f"  Response preview: {response[:100]}...")
                                return True
                        break
                
                print(f"{Fore.YELLOW}⚠ No assistant response found{Style.RESET_ALL}")
                return False
            else:
                print(f"{Fore.RED}✗ Run failed with status: {run.status}{Style.RESET_ALL}")
                if hasattr(run, 'last_error') and run.last_error:
                    print(f"{Fore.RED}  Error: {run.last_error}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}✗ Query test failed: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_user_lookup(self) -> bool:
        """Test user lookup functionality."""
        print(f"\n{Fore.CYAN}Testing user lookup...{Style.RESET_ALL}")
        
        if not self.project_client or not self.agent or not self.thread:
            print(f"{Fore.RED}✗ Prerequisites not met{Style.RESET_ALL}")
            return False
        
        try:
            query = "Can you find user information for noah.brown7922@example.com?"
            print(f"  Query: '{query}'")
            
            # Add message
            self.project_client.agents.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=query
            )
            
            # Run agent with auto-approve handler
            run_handler = AutoApproveRunHandler()
            run = self.project_client.agents.runs.create_and_process(
                thread_id=self.thread.id,
                agent_id=self.agent.id,
                run_handler=run_handler
            )
            
            if run.status == "completed":
                # Check for tool calls
                run_steps = self.project_client.agents.run_steps.list(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
                
                tool_called = False
                for step in run_steps:
                    if hasattr(step, 'step_details') and step.step_details:
                        if hasattr(step.step_details, 'tool_calls') and step.step_details.tool_calls:
                            tool_called = True
                            break
                
                if tool_called:
                    print(f"{Fore.GREEN}✓ User lookup successful (tool was called){Style.RESET_ALL}")
                    return True
                else:
                    print(f"{Fore.YELLOW}⚠ No tool calls detected{Style.RESET_ALL}")
                    return False
            else:
                print(f"{Fore.RED}✗ Run failed with status: {run.status}{Style.RESET_ALL}")
                if hasattr(run, 'last_error') and run.last_error:
                    print(f"{Fore.RED}  Error: {run.last_error}{Style.RESET_ALL}")
                return False
                
        except Exception as e:
            print(f"{Fore.RED}✗ User lookup test failed: {e}{Style.RESET_ALL}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        print(f"\n{Fore.CYAN}Cleaning up resources...{Style.RESET_ALL}")
        
        try:
            if self.agent and self.project_client:
                self.project_client.agents.delete_agent(self.agent.id)
                print(f"{Fore.GREEN}✓ Deleted agent{Style.RESET_ALL}")
            
            if self.thread and self.project_client:
                self.project_client.agents.threads.delete(self.thread.id)
                print(f"{Fore.GREEN}✓ Deleted thread{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠ Cleanup warning: {e}{Style.RESET_ALL}")
    
    def run_all_tests(self, notebook_mode: bool = False) -> dict:
        """Run all retail agent tests.
        
        Args:
            notebook_mode: If True, formats output for notebook display
        """
        if notebook_mode:
            print("=" * 70)
            print("🤖 Retail Agent Test Suite")
            print("=" * 70)
            print(f"Model: {self.model_name}")
            print(f"MCP Server: {self.mcp_server_url}\n")
        else:
            print(f"{Fore.YELLOW}{'='*60}")
            print(f"Retail Agent Test Suite")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}{Style.RESET_ALL}\n")
        
        # Run tests in sequence
        self.results['connection'] = self.test_connection()
        
        if self.results['connection']:
            self.results['agent_creation'] = self.test_agent_creation()
            self.results['thread_creation'] = self.test_thread_creation()
            
            if self.results.get('agent_creation') and self.results.get('thread_creation'):
                self.results['simple_query'] = self.test_simple_query()
                self.results['user_lookup'] = self.test_user_lookup()
            else:
                self.results['simple_query'] = False
                self.results['user_lookup'] = False
        else:
            self.results['agent_creation'] = False
            self.results['thread_creation'] = False
            self.results['simple_query'] = False
            self.results['user_lookup'] = False
        
        # Cleanup
        self.cleanup()
        
        # Summary
        if notebook_mode:
            print("\n" + "=" * 70)
            print("📊 Test Summary")
            print("=" * 70)
            
            passed = sum(1 for v in self.results.values() if v)
            total = len(self.results)
            
            # Create formatted table
            print(f"\n{'Test Name':<30} {'Status':<15} {'Result'}")
            print("-" * 70)
            
            for test_name, result in self.results.items():
                status_icon = "✅" if result else "❌"
                status_text = "PASS" if result else "FAIL"
                formatted_name = test_name.replace('_', ' ').title()
                print(f"{formatted_name:<30} {status_text:<15} {status_icon}")
            
            print("-" * 70)
            print(f"\n📈 Results: {passed}/{total} tests passed ({passed*100//total if total > 0 else 0}%)")
            
            if passed == total:
                print("\n✅ All tests passed! Retail agent is fully operational.")
            elif passed > 0:
                print(f"\n⚠️  Some tests failed. Check configuration and permissions.")
            else:
                print("\n❌ All tests failed. Check Azure credentials and configuration.")
        else:
            print(f"\n{Fore.YELLOW}{'='*60}")
            print("Test Summary")
            print(f"{'='*60}{Style.RESET_ALL}")
            
            passed = sum(1 for v in self.results.values() if v)
            total = len(self.results)
            
            for test_name, result in self.results.items():
                status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if result else f"{Fore.RED}FAIL{Style.RESET_ALL}"
                print(f"  {test_name:20s}: {status}")
            
            print(f"\n{Fore.YELLOW}Total: {passed}/{total} tests passed{Style.RESET_ALL}")
            
            if passed == total:
                print(f"{Fore.GREEN}✓ All tests passed!{Style.RESET_ALL}")
            elif passed > 0:
                print(f"{Fore.YELLOW}⚠ Some tests failed{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ All tests failed{Style.RESET_ALL}")
        
        return self.results


def quick_test() -> bool:
    """Quick test to check if configuration is valid."""
    connection_string = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING")
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    
    if not connection_string:
        print(f"{Fore.RED}Missing AZURE_AI_PROJECT_CONNECTION_STRING{Style.RESET_ALL}")
        return False
    
    if not mcp_server_url:
        print(f"{Fore.RED}Missing MCP_SERVER_URL{Style.RESET_ALL}")
        return False
    
    return True


def main():
    """Run the test suite from command line."""
    import sys
    
    print(f"{Fore.CYAN}Retail Agent Test Suite{Style.RESET_ALL}\n")
    
    if not quick_test():
        print(f"\n{Fore.RED}Configuration check failed. Please set required environment variables.{Style.RESET_ALL}")
        sys.exit(1)
    
    tester = RetailAgentTester()
    results = tester.run_all_tests(notebook_mode=False)
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

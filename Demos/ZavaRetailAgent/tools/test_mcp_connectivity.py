"""
MCP Server Connectivity Test Utility

Tests basic connectivity and functionality of the Retail MCP Server.
Can be imported and used from Jupyter notebooks or run standalone.
"""

import requests
import json
from typing import Dict, Any, Optional
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Default MCP server URL
DEFAULT_MCP_SERVER = "https://retail-mcp-server-sim.braveflower-06b407cc.eastus.azurecontainerapps.io"


class MCPConnectivityTester:
    """Test connectivity and basic operations against the Retail MCP Server."""
    
    def __init__(self, base_url: str = DEFAULT_MCP_SERVER):
        """Initialize the tester with the MCP server base URL."""
        self.base_url = base_url.rstrip('/')
        self.results = {}
    
    def test_health_check(self) -> bool:
        """Test the /health endpoint."""
        print(f"{Fore.CYAN}Testing health check endpoint...{Style.RESET_ALL}")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            success = response.status_code == 200
            
            if success:
                print(f"{Fore.GREEN}✓ Health check passed{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
                try:
                    print(f"  Response: {response.json()}")
                except:
                    print(f"  Response: {response.text}")
            else:
                print(f"{Fore.RED}✗ Health check failed{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
            
            self.results['health'] = success
            return success
            
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}✗ Health check failed with error: {e}{Style.RESET_ALL}")
            self.results['health'] = False
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test the root / endpoint (documentation page)."""
        print(f"\n{Fore.CYAN}Testing root endpoint...{Style.RESET_ALL}")
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            success = response.status_code == 200
            
            if success:
                print(f"{Fore.GREEN}✓ Root endpoint accessible{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
                print(f"  Content length: {len(response.text)} bytes")
            else:
                print(f"{Fore.RED}✗ Root endpoint failed{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
            
            self.results['root'] = success
            return success
            
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}✗ Root endpoint failed with error: {e}{Style.RESET_ALL}")
            self.results['root'] = False
            return False
    
    def test_tools_endpoint_basic(self) -> bool:
        """Test the /tools endpoint with a basic function call."""
        print(f"\n{Fore.CYAN}Testing /tools endpoint (OpenAI compatible)...{Style.RESET_ALL}")
        try:
            payload = {
                "type": "function_call",
                "id": "fc_test_001",
                "call_id": "call_test_001",
                "name": "list_all_product_types",
                "arguments": "{}",
                "trace_id": "trace_test_001"
            }
            
            response = requests.post(
                f"{self.base_url}/tools",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            success = response.status_code == 200
            
            if success:
                print(f"{Fore.GREEN}✓ Tools endpoint working{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
                result = response.json()
                print(f"  Response type: {result.get('type')}")
                print(f"  Call ID: {result.get('call_id')}")
                
                # Try to parse the output
                try:
                    output = json.loads(result.get('output', '{}'))
                    if isinstance(output, list):
                        print(f"  Products returned: {len(output)} types")
                    else:
                        print(f"  Output: {output}")
                except:
                    print(f"  Raw output: {result.get('output', 'N/A')[:200]}")
            else:
                print(f"{Fore.RED}✗ Tools endpoint failed{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
            
            self.results['tools_basic'] = success
            return success
            
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}✗ Tools endpoint failed with error: {e}{Style.RESET_ALL}")
            self.results['tools_basic'] = False
            return False
    
    def test_direct_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict]:
        """Test a direct tool call via /tools endpoint using OpenAI format."""
        print(f"\n{Fore.CYAN}Testing direct tool call: {tool_name}...{Style.RESET_ALL}")
        try:
            # Use OpenAI function calling compatible format
            payload = {
                "type": "function_call",
                "id": f"fc_{tool_name}",
                "call_id": f"call_{tool_name}",
                "name": tool_name,
                "arguments": json.dumps(arguments),
                "trace_id": f"trace_{tool_name}"
            }
            
            response = requests.post(
                f"{self.base_url}/tools",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            success = response.status_code == 200
            
            if success:
                print(f"{Fore.GREEN}✓ Tool call successful{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
                response_data = response.json()
                
                # Parse the output field which contains stringified JSON
                try:
                    result = json.loads(response_data.get('output', '{}'))
                except:
                    result = response_data.get('output', {})
                
                # Display result summary
                if isinstance(result, list):
                    print(f"  Result: List with {len(result)} items")
                elif isinstance(result, dict):
                    print(f"  Result keys: {list(result.keys())}")
                else:
                    print(f"  Result: {str(result)[:200]}")
                
                return result
            else:
                print(f"{Fore.RED}✗ Tool call failed{Style.RESET_ALL}")
                print(f"  Status: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"{Fore.RED}✗ Tool call failed with error: {e}{Style.RESET_ALL}")
            return None
    
    def test_user_lookup(self) -> bool:
        """Test user lookup by email."""
        print(f"\n{Fore.CYAN}Testing user lookup by email...{Style.RESET_ALL}")
        
        result = self.test_direct_tool_call(
            "find_user_id_by_email",
            {"email": "noah.brown7922@example.com"}
        )
        
        # Result can be either a string user_id or a dict with user_id key
        success = result is not None and (isinstance(result, str) or 'user_id' in result)
        self.results['user_lookup'] = success
        
        if success:
            print(f"{Fore.GREEN}✓ User lookup successful{Style.RESET_ALL}")
            user_id = result if isinstance(result, str) else result.get('user_id')
            print(f"  Found user_id: {user_id}")
        
        return success
    
    def test_product_list(self) -> bool:
        """Test listing all product types."""
        print(f"\n{Fore.CYAN}Testing product listing...{Style.RESET_ALL}")
        
        result = self.test_direct_tool_call(
            "list_all_product_types",
            {}
        )
        
        # Result can be either a list or a dict of product types
        success = result is not None and (isinstance(result, (list, dict)))
        self.results['product_list'] = success
        
        if success:
            print(f"{Fore.GREEN}✓ Product listing successful{Style.RESET_ALL}")
            if isinstance(result, dict):
                print(f"  Found {len(result)} product types")
                print(f"  Sample products: {list(result.keys())[:3]}")
            else:
                print(f"  Found {len(result)} product types")
                if len(result) > 0:
                    print(f"  Sample products: {result[:3]}")
        
        return success
    
    def run_all_tests(self, notebook_mode: bool = False) -> Dict[str, bool]:
        """Run all connectivity tests.
        
        Args:
            notebook_mode: If True, formats output for notebook display with HTML-style formatting
        """
        if notebook_mode:
            # Notebook-friendly header
            print("=" * 70)
            print("🔌 MCP Server Connectivity Test Suite")
            print("=" * 70)
            print(f"Server: {self.base_url}\n")
        else:
            print(f"{Fore.YELLOW}{'='*60}")
            print(f"MCP Server Connectivity Test Suite")
            print(f"Server: {self.base_url}")
            print(f"{'='*60}{Style.RESET_ALL}\n")
        
        # Run all tests
        self.test_health_check()
        self.test_root_endpoint()
        self.test_tools_endpoint_basic()
        self.test_user_lookup()
        self.test_product_list()
        
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
            print(f"\n📈 Results: {passed}/{total} tests passed ({passed*100//total}%)")
            
            if passed == total:
                print("\n✅ All tests passed! MCP server is fully operational.")
            elif passed > 0:
                print(f"\n⚠️  Some tests failed. Server may have limited functionality.")
            else:
                print("\n❌ All tests failed. Server may be unreachable.")
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
                print(f"{Fore.GREEN}✓ All tests passed! MCP server is fully operational.{Style.RESET_ALL}")
            elif passed > 0:
                print(f"{Fore.YELLOW}⚠ Some tests failed. Server may have limited functionality.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}✗ All tests failed. Server may be unreachable.{Style.RESET_ALL}")
        
        return self.results


def quick_test(base_url: str = DEFAULT_MCP_SERVER) -> bool:
    """
    Quick connectivity test - just checks if the server is reachable.
    Returns True if server is accessible, False otherwise.
    """
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def main():
    """Run the test suite from command line."""
    import sys
    
    # Allow custom URL as command line argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MCP_SERVER
    
    tester = MCPConnectivityTester(base_url)
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

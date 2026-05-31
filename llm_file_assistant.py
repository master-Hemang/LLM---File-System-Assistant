import os
import json
from pathlib import Path
from typing import List, Dict, Any
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Gemini library not installed. Run: pip install google-generativeai")

from fs_tools import FileSystemTools


class GeminiFileAssistant:
    """LLM-powered file assistant using Google Gemini with tool calling"""
    
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash-lite"):
        """
        Initialize the assistant with Gemini
        
        Args:
            api_key: Google API key (get from https://makersuite.google.com/app/apikey)
            model: Gemini model to use (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Gemini library required. Run: pip install google-generativeai")
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required.\n"
                "Get your free API key from: https://makersuite.google.com/app/apikey\n"
                "Then set: export GEMINI_API_KEY='your-key-here'"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.tools = FileSystemTools()
        
        # Keep conversation history
        self.chat_history = []
    
    def extract_tool_calls(self, response_text: str) -> List[Dict]:
        """
        Extract tool calls from Gemini's response
        
        Looks for JSON objects in the response that match tool call format
        """
        tool_calls = []
        
        # Try to parse entire response as JSON first
        try:
            data = json.loads(response_text.strip())
            if "tool" in data and "params" in data:
                return [data]
            elif isinstance(data, list):
                for item in data:
                    if "tool" in item and "params" in item:
                        tool_calls.append(item)
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError:
            pass
        
        # Pattern to find JSON objects (more flexible pattern)
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"params"\s*:\s*\{[^{}]*\}\s*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                # Clean up the match
                match = match.strip()
                tool_call = json.loads(match)
                if "tool" in tool_call and "params" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    def execute_tool(self, tool_call: Dict) -> Dict:
        """Execute a tool and return the result"""
        tool_name = tool_call.get("tool")
        params = tool_call.get("params", {})
        
        print(f"   🔧 Executing: {tool_name}({params})")
        
        if tool_name == "list_files":
            return self.tools.list_files(**params)
        elif tool_name == "read_file":
            return self.tools.read_file(**params)
        elif tool_name == "search_in_file":
            return self.tools.search_in_file(**params)
        elif tool_name == "write_file":
            return self.tools.write_file(**params)
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def process_query(self, user_query: str, verbose: bool = False) -> str:
        """
        Process a natural language query using Gemini
        
        Args:
            user_query: User's natural language request
            verbose: If True, print tool calls as they happen
            
        Returns:
            Assistant's response with results
        """
        # Build the prompt with tool descriptions - simplified for better JSON response
        system_prompt = f"""You are a file assistant that helps users manage files. You have access to these tools:

1. list_files - List files in a directory
   Format: {{"tool": "list_files", "params": {{"directory": "path", "extension": ".pdf"}}}}
   Example: {{"tool": "list_files", "params": {{"directory": "./sample_data"}}}}

2. read_file - Read a file's content
   Format: {{"tool": "read_file", "params": {{"filepath": "path/to/file"}}}}

3. search_in_file - Search for text in a file
   Format: {{"tool": "search_in_file", "params": {{"filepath": "path/to/file", "keyword": "search term"}}}}

4. write_file - Write content to a file
   Format: {{"tool": "write_file", "params": {{"filepath": "path/to/file", "content": "text to write"}}}}

CRITICAL INSTRUCTIONS:
- When you need to perform an action, respond ONLY with the JSON object (no other text)
- Use EXACTLY the format shown above
- For searching, use the exact keyword provided
- If the user asks about "resumes folder", use "./sample_data" as the directory
- Respond with valid JSON only - no explanations, no markdown

User request: {user_query}

Previous conversation:
{self._format_history()}

Respond with tool JSON now:"""

        try:
            # Get response from Gemini
            response = self.model.generate_content(system_prompt)
            response_text = response.text.strip()
            
            if verbose:
                print(f"\n📝 Raw response: {response_text}")
            
            # Extract and execute tool calls
            tool_calls = self.extract_tool_calls(response_text)
            
            if tool_calls:
                print(f"\n🔧 Found {len(tool_calls)} tool call(s) to execute...")
                
                results = []
                for tool_call in tool_calls:
                    result = self.execute_tool(tool_call)
                    results.append({
                        "tool": tool_call['tool'],
                        "params": tool_call['params'],
                        "result": result
                    })
                
                # Store in history
                self.chat_history.append({
                    "user": user_query,
                    "assistant": response_text,
                    "tool_results": results
                })
                
                # Format the results for the user
                return self._format_results(results, user_query)
            else:
                # If no tool calls detected, return the response as is
                return response_text
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _format_results(self, results: List[Dict], original_query: str) -> str:
        """Format tool results into a user-friendly response"""
        output = []
        
        for item in results:
            tool = item['tool']
            result = item['result']
            
            if tool == "list_files":
                if result.get("success"):
                    files = result.get("files", [])
                    if files:
                        output.append(f"📁 Found {len(files)} file(s):")
                        for f in files:
                            output.append(f"   • {f['name']} ({f.get('size_kb', 0)} KB)")
                    else:
                        output.append("📁 No files found matching the criteria")
                else:
                    output.append(f"❌ Error listing files: {result.get('error')}")
            
            elif tool == "search_in_file":
                if result.get("success"):
                    matches = result.get("matches", [])
                    if matches:
                        output.append(f"🔍 Found {len(matches)} match(es) in {result.get('filepath', 'file')}:")
                        for match in matches[:3]:  # Show first 3 matches
                            output.append(f"   Line {match.get('line_number')}: ...{match.get('context')}...")
                        if len(matches) > 3:
                            output.append(f"   ... and {len(matches) - 3} more")
                    else:
                        output.append(f"🔍 No matches found for '{result.get('keyword')}'")
                else:
                    output.append(f"❌ Error searching: {result.get('error')}")
            
            elif tool == "read_file":
                if result.get("success"):
                    content = result.get("content", "")
                    # Show first 500 characters
                    preview = content[:500] + "..." if len(content) > 500 else content
                    output.append(f"📄 Content of {result.get('metadata', {}).get('filename', 'file')}:\n{preview}")
                else:
                    output.append(f"❌ Error reading file: {result.get('error')}")
            
            elif tool == "write_file":
                if result.get("success"):
                    output.append(f"✅ {result.get('message')}")
                else:
                    output.append(f"❌ Error writing file: {result.get('error')}")
        
        return "\n".join(output) if output else "No results to display"
    
    def _format_history(self) -> str:
        """Format chat history for context"""
        if not self.chat_history:
            return "No previous conversation."
        
        formatted = []
        for entry in self.chat_history[-3:]:  # Last 3 exchanges
            formatted.append(f"User: {entry['user']}")
            formatted.append(f"Assistant used: {[t['tool'] for t in entry.get('tool_results', [])]}")
        return "\n".join(formatted)
    
    def interactive_mode(self):
        """Run the assistant in interactive command-line mode"""
        print("\n" + "="*70)
        print("🤖 Gemini File Assistant - Interactive Mode (FREE TIER)")
        print("="*70)
        print("I can help you read, search, and manage your resume files.")
        print("\n📝 Example queries:")
        print("   • 'List all PDF files in the sample_data folder'")
        print("   • 'Find resumes mentioning Python'")
        print("   • 'Create a summary file for resume_john_doe.pdf'")
        print("   • 'Search for leadership experience in all text files'")
        print("\n💡 Type 'quit' to exit, 'help' for more examples")
        print("="*70 + "\n")
        
        while True:
            try:
                query = input("📝 You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if not query:
                    continue
                
                print("\n🤖 Assistant: Processing...")
                response = self.process_query(query, verbose=True)
                print(f"\n🤖 Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")
    
    def _show_help(self):
        """Show example queries"""
        help_text = """
📚 EXAMPLE QUERIES:

📁 FILE LISTING:
   • "List all files in the sample_data folder"
   • "List all PDF files in sample_data"
   • "Show me only text files"

🔍 SEARCHING:
   • "Find resumes mentioning Python"
   • "Search for 'machine learning' in all files"
   • "Which files contain the word 'leadership'?"

📖 READING FILES:
   • "Read the file sample_data/resume1.txt"
   • "Show me the contents of the first resume"

✍️ WRITING FILES:
   • "Create a summary file called analysis.txt"
   • "Write a list of all resume filenames to summary.txt"

💡 TIPS:
   • Use "sample_data" as the folder name (not "resumes")
   • Be specific about file extensions when filtering
"""
        print(help_text)


def list_available_models():
    """Helper function to list all available Gemini models"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY environment variable not set")
            return
        
        genai.configure(api_key=api_key)
        print("\n📋 Available Gemini models:\n")
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"   ✅ {model.name}")
        print("\n")
    except Exception as e:
        print(f"Error listing models: {e}")


def quick_test():
    """Quick test function to verify tools work without LLM"""
    print("Testing FileSystemTools...")
    tools = FileSystemTools()
    
    # Test list files
    result = tools.list_files("./sample_data" if os.path.exists("./sample_data") else ".")
    if result["success"]:
        print(f"✅ Found {result['count']} files in directory")
        for f in result.get("files", [])[:5]:
            print(f"   • {f['name']}")
    else:
        print(f"❌ Error: {result['error']}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini File Assistant - Natural language file operations")
    parser.add_argument("--query", type=str, help="Single query to process and exit")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", 
                       help="Gemini model (gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro)")
    parser.add_argument("--test", action="store_true", help="Run quick test without LLM")
    parser.add_argument("--list-models", action="store_true", help="List all available Gemini models")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    if args.test:
        quick_test()
        return
    
    try:
        assistant = GeminiFileAssistant(model=args.model)
        
        if args.query:
            # Single query mode
            print(f"Query: {args.query}\n")
            response = assistant.process_query(args.query, verbose=True)
            print(f"\nResponse: {response}")
        else:
            # Interactive mode
            assistant.interactive_mode()
            
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n🔧 To fix:")
        print("   1. Go to https://makersuite.google.com/app/apikey")
        print("   2. Click 'Create API Key'")
        print("   3. Copy your key")
        print("   4. Set environment variable: export GEMINI_API_KEY='your-key-here'")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()  
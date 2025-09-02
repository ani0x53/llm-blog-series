"""
LangGraph Agent - Wrapping your working agent with LangGraph
==========================================================

This keeps your working SimpleAgent exactly as-is and just adds LangGraph
state management and workflow on top of it.
"""

import requests
import json
import os
import re
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# Configuration - Same as your original
MODEL_NAME = "llama3.2:1b"

def call_ollama(prompt, model=MODEL_NAME):
    """Call local Ollama - Your exact function"""
    try:
        response = requests.post('http://localhost:11434/api/generate', json={
            'model': model, 'prompt': prompt, 'stream': False
        })
        result = response.json()
        
        if 'response' in result:
            return result['response']
        else:
            return f"Ollama response missing 'response' key: {result}"
            
    except Exception as e:
        return f"Ollama error: {e}"

def calculate(expr):
    """Calculator tool - Your exact function"""
    try:
        return f"{expr} = {eval(expr)}"
    except:
        return f"Math error with: {expr}"

def list_files(directory="."):
    """List files tool - Your exact function"""
    try:
        files = os.listdir(directory)
        return f"Files in {directory}: {', '.join(files[:10])}"
    except:
        return f"Can't list files in {directory}"

def write_file(filename, content):
    """Write file tool - Your exact function"""
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Wrote to {filename}"
    except:
        return f"Can't write to {filename}"

def read_file(filename):
    """Read file tool - Your exact function"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return f"Content of {filename}: {content[:200]}..."
    except:
        return f"Can't read {filename}"

# Your exact SimpleAgent class
class SimpleAgent:
    def __init__(self):
        self.history = []
        self.tools = {
            'calculate': calculate,
            'list_files': list_files,
            'write_file': write_file,
            'read_file': read_file
        }
    
    def chat(self, user_input):
        """Main chat function - Your exact code"""
        
        # Build prompt with history and tools
        history_text = ""
        for h in self.history[-5:]:  # Last 5 exchanges
            history_text += f"User: {h['user']}\nAgent: {h['agent']}\n"
        
        prompt = f"""You are a helpful AI assistant with access to tools.

Available tools:
- calculate(expression) - for math calculations only
- list_files(directory) - when user asks to see/list files  
- write_file(filename, content) - when user asks to write/save to a file
- read_file(filename) - when user asks to read a specific file

IMPORTANT: Only use tools when the user explicitly asks for them.
- For greetings (hi, hello, hey) - just respond normally
- For general questions - just respond normally  
- For math questions - use calculate tool
- For file requests - use appropriate file tool

When you need a tool, use this exact format:
TOOL: tool_name
ARGS: arguments

For write_file, use: filename|content

Previous conversation:
{history_text}

User: {user_input}
Assistant:"""

        response = call_ollama(prompt)
        
        # Check for tool use
        if "TOOL:" in response and "ARGS:" in response:
            final_response = self._use_tool(response, user_input)
        else:
            final_response = response
        
        # Save to history
        self.history.append({'user': user_input, 'agent': final_response})
        return final_response
    
    def _use_tool(self, response, original_question):
        """Execute tools - Your exact code"""
        try:
            tool_name = re.search(r'TOOL:\s*(\w+)', response).group(1)
            tool_args = re.search(r'ARGS:\s*(.+)', response).group(1).strip()
            
            print(f"🔧 Using {tool_name} with: {tool_args}")
            
            if tool_name == 'write_file' and '|' in tool_args:
                filename, content = tool_args.split('|', 1)
                result = self.tools[tool_name](filename.strip(), content.strip())
            elif tool_name in self.tools:
                result = self.tools[tool_name](tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Get final response
            final_prompt = f"""User asked: {original_question}
Tool {tool_name} returned: {result}
Give a helpful response:"""
            
            return call_ollama(final_prompt)
            
        except Exception as e:
            return f"Tool error: {e}"

# LangGraph State - Just for managing conversation flow
class ConversationState(TypedDict):
    messages: Annotated[list[str], operator.add]
    user_input: str
    response: str

# LangGraph wrapper around your SimpleAgent
class LangGraphAgent:
    def __init__(self):
        # Your working agent
        self.simple_agent = SimpleAgent()
        
        # LangGraph workflow
        self.app = self._create_workflow()
    
    def _create_workflow(self):
        """Create LangGraph workflow that uses your SimpleAgent"""
        
        def process_user_input(state):
            """Process user input using your SimpleAgent"""
            user_input = state["user_input"]
            response = self.simple_agent.chat(user_input)
            
            return {
                "messages": [f"User: {user_input}", f"Agent: {response}"],
                "response": response
            }
        
        # Create workflow
        workflow = StateGraph(ConversationState)
        workflow.add_node("chat", process_user_input)
        workflow.set_entry_point("chat")
        workflow.add_edge("chat", END)
        
        return workflow.compile()
    
    def chat(self, user_input: str) -> str:
        """Chat using LangGraph workflow"""
        try:
            result = self.app.invoke({
                "user_input": user_input,
                "messages": [],
                "response": ""
            })
            return result["response"]
        except Exception as e:
            return f"Error: {e}"

def main():
    """Interactive demo with LangGraph layer"""
    print("🤖 LangGraph AI Agent (Wrapping Your Working Code)")
    print("Same functionality, now with LangGraph state management!")
    print("Try: math, file operations, or just chat!")
    print("Type 'quit' to exit\n")
    
    try:
        # Create LangGraph agent (which uses your SimpleAgent internally)
        agent = LangGraphAgent()
        print("✅ LangGraph wrapper initialized!\n")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Bye!")
            break
            
        if user_input:
            print("🤔 Thinking...")
            response = agent.chat(user_input)
            print(f"Agent: {response}\n")

if __name__ == "__main__":
    main()
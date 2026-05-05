import os
import json
from groq import Groq
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Tools ────────────────────────────────────────────────

def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results:
                return "No results found."
            output = ""
            for r in results:
                output += f"Title: {r['title']}\n"
                output += f"Summary: {r['body']}\n\n"
            return output.strip()
    except Exception as e:
        return f"Search error: {str(e)}"

def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "pounds"): 2.20462,
        ("pounds", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
    }

    key = (from_unit.lower(), to_unit.lower())

    if key not in conversions:
        return f"Cannot convert {from_unit} to {to_unit}"

    conversion = conversions[key]

    if callable(conversion):
        result = conversion(value)
    else:
        result = value * conversion

    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"

# ── Tool definitions for LLM ─────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Useful for math calculations, arithmetic, percentages, and any numerical computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate. Example: '15/100 * 2847'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Useful for finding current information, facts, news, people, companies, or anything that requires looking up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Converts units of measurement. Use for converting km to miles, kg to pounds, celsius to fahrenheit, meters to feet and vice versa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numerical value to convert"
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The unit to convert from. Example: km, miles, kg, pounds, celsius, fahrenheit"
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The unit to convert to."
                    }
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]

# ── Agent loop ───────────────────────────────────────────

def run_agent(user_question: str):
    print(f"\nUser: {user_question}")
    print("-" * 40)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to three tools: "
                "a calculator for math, a search engine for facts, "
                "and a unit converter for unit conversions. "
                "Always use the appropriate tool to answer accurately. "
                "Think step by step before answering."
            )
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    # Step 1 — Ask LLM which tool to use
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0
    )

    message = response.choices[0].message

    # Step 2 — If LLM wants to use a tool
    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"Agent thinks: using {tool_name}")
            print(f"Input: {tool_args}")

            # Step 3 — Execute the tool
            if tool_name == "calculator":
                result = calculator(tool_args["expression"])
            elif tool_name == "search":
                result = search(tool_args["query"])
            elif tool_name == "unit_converter":
                result = unit_converter(
                    tool_args["value"],
                    tool_args["from_unit"],
                    tool_args["to_unit"]
                )
            else:
                result = "Unknown tool"

            print(f"Tool result: {result[:200]}")
            print("-" * 40)

            # Step 4 — Send result back to LLM
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message.tool_calls
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Step 5 — Get final answer from LLM
        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.0
        )
        final_answer = final_response.choices[0].message.content

    else:
        final_answer = message.content

    print(f"Agent: {final_answer}\n")
    return final_answer

# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("AI Agent with Tool Use")
    print("Tools: calculator + search + unit converter")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "quit":
            break
        if question:
            run_agent(question)
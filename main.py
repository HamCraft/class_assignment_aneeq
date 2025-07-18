from dataclasses import dataclass
import random
import asyncio

from agents import Agent, OpenAIChatCompletionsModel, RunContextWrapper, Runner, TResponseInputItem, function_tool, handoff

# OpenAI Agents SDK Setup
import os
from agents import set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
#Reference: https://ai.google.dev/gemini-api/docs/openai
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_tracing_disabled(disabled=True)


@dataclass
class UserProfile:
    id: str
    name: str
    shopping_cart: list[str]

@function_tool
async def get_budget(wrapper: RunContextWrapper[UserProfile]):
    """
    Get the account balance of the user using the user's id and their linked bank account
    """
    print("Getting account balance")
    user_id = wrapper.context.id

    # pretend we are fetching the account balance from a database

    return 100.0

@function_tool
async def search_for_item(wrapper: RunContextWrapper[UserProfile], item: str) -> str:
    """
    Search for an item in the database
    """
    print("Searching for item")
    # randomly generate a price for the item
    price = random.randint(1, 100)
    return f"Found {item} in the database for ${price}.00"

@function_tool
async def get_shopping_cart(wrapper: RunContextWrapper[UserProfile]) -> list[str]:
    print("Getting shopping cart")
    return wrapper.context.shopping_cart

@function_tool
async def add_to_shopping_cart(wrapper: RunContextWrapper[UserProfile], items: list[str]) -> None:
    print("Adding items to shopping cart")
    wrapper.context.shopping_cart.extend(items)
    
@function_tool
async def purchase_items(wrapper: RunContextWrapper[UserProfile]) -> None:
    print("Purchasing items")
    
    # we could take the items from the shopping cart and purchase them using some external API
    # for now, we'll just print a message
    print(f"Successfully purchased items: {wrapper.context.shopping_cart}")


customer_agent = Agent(
    name="Customer Support Agent",
    handoff_description="Specialist agent for handling customer queries",
    instructions="You provide assistance with customer-related queries and issues. If the user has a question about their account or needs help with a purchase, you will assist them.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

def customer_agent_handoff(ctx: RunContextWrapper[None]):
    print("Handing off to customer agent")



shopping_agent = Agent[UserProfile](
    name="Shopping Assistant",
    instructions=(
        "You are a shopping assistant dedicated to helping the user with their grocery shopping needs."
        "Your primary role is to assist in creating a shopping plan that fits within the user's budget."
        "Start by getting the user's budget using the tool get_budget."
        "Provide suggestions for items if requested, and always aim to keep the total cost within the user's budget."
        "If the user is nearing or exceeding their budget, inform them and suggest alternatives or adjustments to the shopping list."
        "If the user authorizes it, you can purchase the items using the tool purchase_items."
    ),
    tools=[get_shopping_cart, add_to_shopping_cart, get_budget, search_for_item, purchase_items],
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    handoffs=[handoff(customer_agent, on_handoff=customer_agent_handoff)],
)

profile = UserProfile(id="123", name="Alex", shopping_cart=[])

print("You are now chatting with the shopping assistant. Type 'exit' to end the conversation.")

async def main():
    convo_items: list[TResponseInputItem] = []
    while True:
        user_input = input("You: ")

        if user_input == "exit":
            print("Goodbye!, Come back next time for more shopping assistance.")
            break

        convo_items.append({"content": user_input, "role": "user"})
        result = await Runner.run(shopping_agent, convo_items, context=profile,)
        
        print(f"Shopping Assistant: {result.final_output}")
        
        convo_items = result.to_input_list()

if __name__ == "__main__":
    asyncio.run(main())
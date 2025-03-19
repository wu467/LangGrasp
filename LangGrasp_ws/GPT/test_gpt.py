from openai import OpenAI

client = OpenAI(
    api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    base_url="XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
)

completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "Your identity is the controlling brain of the robot"},
    {"role": "user", "content": "hello"}
  ]
)

print(completion.choices[0].message.content)




import asyncio
import aiohttp
import json
import time

async def send_request(session, prompt):
    start_time = time.time()  # Record the start time
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral:7b-instruct-v0.2-q6_K",
        "prompt": prompt,
        "stream": False
    }
    async with session.post(url, data=json.dumps(payload)) as response:
        try:
            result = await response.json()
        except aiohttp.ContentTypeError:
            result = await response.text()
            print(f"Non-JSON response for prompt '{prompt}': {result}")
            return None
    end_time = time.time()  # Record the end time
    duration = end_time - start_time  # Calculate the duration
    return result, start_time, end_time, duration  # Return result, start time, end time, and duration

async def main():
    prompts = [
        "What is the capital of France?",
        "How many continents are there?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the chemical symbol for water?",
        "Who painted the Mona Lisa?",
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        # Print the results with durations and start/end times for each prompt
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            if result is not None:
                response = result[0]['response']
                duration = result[3]
                start_time = result[1]
                end_time = result[2]
                print(f"Result {i+1} for prompt '{prompt}': {response} (Duration: {duration:.2f} seconds, Start time: {start_time:.2f}, End time: {end_time:.2f})")

    # Print the start and end times for the entire script
    start_time_all = min(result[1] for result in results if result is not None)
    end_time_all = max(result[2] for result in results if result is not None)
    total_execution_time = end_time_all - start_time_all
    print(f"Total execution time (all prompts): {total_execution_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())

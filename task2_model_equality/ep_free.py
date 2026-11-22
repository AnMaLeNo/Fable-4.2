"""Targeted second pass: free-text prompts on selected scored endpoints. python ep_free.py eps(comma) n_prompts"""
import sys, json, time, client
FT = ["Write a short story about a fox who learns to paint.", "Describe a walk through an old city at night.",
      "Write a letter from a grandmother to her grandson who is moving abroad.", "Explain how a bicycle stays upright, in simple words.",
      "Write a dialogue between a baker and a customer on a rainy morning.", "Describe your favourite season and why.",
      "Write a short story about a lost umbrella.", "Give advice to someone starting their first job.",
      "Describe a village market in great detail.", "Write a short story about a robot who wants to learn to cook.",
      "Explain why the sky changes colour at sunset.", "Write a diary entry of a sailor after a storm.",
      "Describe an imaginary animal and its habits.", "Write a short story about two friends who find an old map.",
      "Tell me about the history of bread.", "Write a speech for a school graduation.",
      "Describe a train journey through the mountains.", "Write a short fairy tale about a king who could not sleep.",
      "Explain how to make a good cup of tea, with a little story.", "Write a short story about a musician who lost her violin."]
if __name__ == "__main__":
    eps = [f"ep-{int(i):02d}" for i in sys.argv[1].split(",")]; n = int(sys.argv[2])
    jobs = [dict(ep=e, prompt=FT[k], pid=f"ft{k+1:02d}", rep=0, max_tokens=256, tag="free") for k in range(n) for e in eps]
    b = client.budget(); json.dump({k: v["used"] for k, v in b.items()}, open(f"counters_before_free_{int(time.time())}.json", "w"))
    assert max(b[e]["used"] for e in eps) + n <= 70, "would break the reserve of 30"
    client.run(jobs, parallel=4)
    b = client.budget(); json.dump({k: v["used"] for k, v in b.items()}, open(f"counters_after_free_{int(time.time())}.json", "w"))
    print("done")

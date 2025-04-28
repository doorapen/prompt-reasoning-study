import argparse
from utils import load_prompts

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--n_examples", type=int, default=100)
    args = p.parse_args()

    prompts = load_prompts()
    template = prompts[args.family]
    # TODO: load data, format input, call API, save to results/
    print(f"Loaded template for {args.family}: {template}")

if __name__ == "__main__":
    main()

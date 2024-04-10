import json
import argparse
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

def calculate_scores(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        y_label = []
        y_prob = []
        for sample in data:
            y_label.append(sample[0])
            y_prob.append(0.5*sample[1] + 0.5*sample[2])
        p1, r1, _ = precision_recall_curve(y_label, y_prob)
        ap_score = average_precision_score(y_label, y_prob)
        return ap_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Average Precision Score from result JSON file.")
    parser.add_argument('--result', type=str, required=True, help='Path to the result JSON file.')

    args = parser.parse_args()
    score = calculate_scores(args.result)
    print(f"Average Precision Score: {score}")
import pickle
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic data using a trained CTGAN model')
    parser.add_argument('--model', type=str, default='ctgan_model_small_model.pkl',
                        help='Path to the pickled CTGAN model file')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--output', type=str, default='synthetic_data.csv',
                        help='Output CSV file name')
    
    args = parser.parse_args()
    
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
        
    synthetic_data = model.sample(args.n_samples)
    
    synthetic_data.rename(columns={
        'FL_MMSE': 'MMSE',
        'COMBINED_NE4S': "APOE"
    }, inplace=True)
        
    
    synthetic_data.to_csv(args.output, index=False)
    print(f"Generated {args.n_samples} synthetic samples and saved to {args.output}")
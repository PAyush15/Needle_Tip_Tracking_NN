import torch
import csv

def test(csv_file, device):
    total_mae = 0.0
    total_distance = 0.0
    num_samples = 0
    mae_list = []

    # Open CSV file for reading
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header row

        for row in reader:
            # Extract the 3D target and predicted points from the CSV row
            target_3d_str = row[2].strip()  # 3rd column
            pred_3d_str = row[5].strip()    # 6th column

            # Convert string representations to numerical tuples
            target_3d = tuple(map(float, target_3d_str.strip('()').split(',')))
            pred_3d = tuple(map(float, pred_3d_str.strip('()').split(',')))

            # Convert to torch tensors
            target_3d_tensor = torch.tensor(target_3d, device=device)
            pred_3d_tensor = torch.tensor(pred_3d, device=device)

            # Calculate MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(pred_3d_tensor - target_3d_tensor)).item()
            total_mae += mae
            mae_list.append(mae)

            # Calculate Euclidean distance
            distance = torch.sqrt(torch.sum((pred_3d_tensor - target_3d_tensor) ** 2)).item()
            total_distance += distance

            num_samples += 1

    # Calculate average MAE and standard deviation
    average_mae = total_mae / num_samples
    mae_std = torch.std(torch.tensor(mae_list)).item()
    print(f'Average MAE: {average_mae:.4f}')
    print(f'MAE Standard Deviation: {mae_std:.4f}')

    # Calculate average distance
    average_distance = total_distance / num_samples
    print(f'Average distance: {average_distance:.4f}')
    print()

def main():
    result_csv_file = 'Y3D/Corrected_Final_CSVs/Y3D_Traj_14.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test(result_csv_file, device)

if __name__ == '__main__':
    main()

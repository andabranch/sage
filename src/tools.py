import os
import pandas as pd
import matplotlib.pyplot as plt


def calculate_classes():
    df = pd.read_csv('datasets/GTSRB_dataset/Train.csv')

    class_names = {0: 'Speed limit (20km/h)',
                   1: 'Speed limit (30km/h)',
                   2: 'Speed limit (50km/h)',
                   3: 'Speed limit (60km/h)',
                   4: 'Speed limit (70km/h)',
                   5: 'Speed limit (80km/h)',
                   6: 'End of speed limit (80km/h)',
                   7: 'Speed limit (100km/h)',
                   8: 'Speed limit (120km/h)',
                   9: 'No passing',
                   10: 'No passing veh over 3.5 tons',
                   11: 'Right-of-way at intersection',
                   12: 'Priority road',
                   13: 'Yield',
                   14: 'Stop',
                   15: 'No vehicles',
                   16: 'Veh > 3.5 tons prohibited',
                   17: 'No entry',
                   18: 'General caution',
                   19: 'Dangerous curve left',
                   20: 'Dangerous curve right',
                   21: 'Double curve',
                   22: 'Bumpy road',
                   23: 'Slippery road',
                   24: 'Road narrows on the right',
                   25: 'Road work',
                   26: 'Traffic signals',
                   27: 'Pedestrians',
                   28: 'Children crossing',
                   29: 'Bicycles crossing',
                   30: 'Beware of ice/snow',
                   31: 'Wild animals crossing',
                   32: 'End speed + passing limits',
                   33: 'Turn right ahead',
                   34: 'Turn left ahead',
                   35: 'Ahead only',
                   36: 'Go straight or right',
                   37: 'Go straight or left',
                   38: 'Keep right',
                   39: 'Keep left',
                   40: 'Roundabout mandatory',
                   41: 'End of no passing',
                   42: 'End no passing veh > 3.5 tons'}

    class_counts = {class_names[class_id]: 0 for class_id in class_names}

    # iterate through the DataFrame and count occurrences of each class
    for index, row in df.iterrows():
        class_id = row['ClassId']
        class_name = class_names.get(class_id, 'Unknown')

        # increment the count for the class
        class_counts[class_name] += 1

    # print the class counts
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"{class_name}: {count} images")

    # class_counts_filtered = {
    #     class_name: count for class_name, count in class_counts.items() if count > 0}

    # # Create a bar chart
    # plt.bar(class_counts_filtered.keys(), class_counts_filtered.values())
    # plt.xlabel("Class Names")
    # plt.ylabel("Number of Images")
    # plt.title("Class Distribution")
    # plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    # plt.show()

# Create Belgian Train.csv with all the classes' metadata


def create_csv_belgian():
    BELGIAN_DATASET_PATH = 'datasets/Belgian_dataset/BelgiumTSC_Testing'

    combined_data = pd.DataFrame(columns=[
        'Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'])

    # iterate through all the GT-000xx.csv files
    for belgianClass in range(62):
        class_id = f'{belgianClass:05d}'
        csv_file = os.path.join(BELGIAN_DATASET_PATH,
                                'Testing', class_id, f'GT-{class_id}.csv')

        if os.path.exists(csv_file):
            # read csv
            df = pd.read_csv(csv_file, delimiter=';')
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    combined_data.to_csv('Test.csv', index=False, sep=';')


if __name__ == "__main__":
    # calculate_classes()
    create_csv_belgian()

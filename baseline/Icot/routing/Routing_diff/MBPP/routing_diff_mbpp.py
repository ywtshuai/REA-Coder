import json
from collections import defaultdict

def read_json_file(file_path):

    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def map_to_two_class(data):

    for entry in data:
        original_label = entry['routing_label']
        if original_label == 'Easy':
            entry['routing_label'] = 'Simple'
        else:  
            entry['routing_label'] = 'Complex'
    return data

def process_data(file1_data, file2_data, file1_path, file2_path):

    file1_data = map_to_two_class(file1_data)
    file2_data = map_to_two_class(file2_data)

    all_data = file1_data + file2_data
    task_info = defaultdict(list)
    for entry in all_data:
        task_info[entry['task_id']].append(entry['routing_label'])

    stats = {
        'file1_path': file1_path,
        'file2_path': file2_path,
        'file1_label_count': {},  
        'file2_label_count': {}, 
        'total_stats': {},
        'simple_stats': {},
        'complex_stats': {}
    }


    def count_labels(data):
        counter = {'Simple': 0, 'Complex': 0}
        for entry in data:
            counter[entry['routing_label']] += 1
        return counter

    stats['file1_label_count'] = count_labels(file1_data)
    stats['file2_label_count'] = count_labels(file2_data)


    same_label_count = 0
    diff_label_count = 0
    total_pairs = 0

    for task_id, labels in task_info.items():
        if len(labels) > 1:
            total_pairs += 1
            if labels[0] == labels[1]:
                same_label_count += 1
            else:
                diff_label_count += 1

    stats['total_stats'] = {
        'total_tasks': len(task_info),
        'same_label_pairs': same_label_count,
        'same_label_percentage': round(same_label_count / total_pairs * 100, 2) if total_pairs else 0,
        'diff_label_pairs': diff_label_count,
        'diff_label_percentage': round(diff_label_count / total_pairs * 100, 2) if total_pairs else 0
    }

    def calculate_label_stats(label):

        file1_count = sum(1 for entry in file1_data if entry['routing_label'] == label)
        file2_count = sum(1 for entry in file2_data if entry['routing_label'] == label)

        file1_tasks = {entry['task_id'] for entry in file1_data}
        file2_tasks = {entry['task_id'] for entry in file2_data}
        union_tasks = file1_tasks.union(file2_tasks)

        one_file_count = 0
        for task_id in union_tasks:
            file1_label = next((entry['routing_label'] for entry in file1_data if entry['task_id'] == task_id), None)
            file2_label = next((entry['routing_label'] for entry in file2_data if entry['task_id'] == task_id), None)
            if (file1_label == label and file2_label != label) or (file2_label == label and file1_label != label):
                one_file_count += 1

        return {
            f'file1_{label.lower()}_count': file1_count,
            f'file2_{label.lower()}_count': file2_count,
            f'union_task_count': len(union_tasks),
            f'one_file_{label.lower()}_count': one_file_count,
            f'one_file_{label.lower()}_percentage': round(one_file_count / len(union_tasks) * 100, 2) if union_tasks else 0
        }

    stats['simple_stats'] = calculate_label_stats('Simple')
    stats['complex_stats'] = calculate_label_stats('Complex')

    return stats

def save_stats_to_json(stats, output_file):

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)

def main():
    file1_path = ""
    file2_path = ""
    output_path = ""

    try:
        file1_data = read_json_file(file1_path)
        file2_data = read_json_file(file2_path)

        stats = process_data(file1_data, file2_data, file1_path, file2_path)
        save_stats_to_json(stats, output_path)

        print(f" {output_path}")
    except FileNotFoundError as e:
        print(f" {str(e)}")
    except json.JSONDecodeError as e:
        print(f" {str(e)}")
    except Exception as e:
        print(f" {str(e)}")

if __name__ == "__main__":
    main()
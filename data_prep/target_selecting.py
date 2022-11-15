
import jsonlines
import os
import csv

def get_question(file_path, question_id_list):
    results = []
    with jsonlines.open(file_path, 'r') as inf:
            for i, item in enumerate(inf):
                question_id = item["id"]
                if question_id in question_id_list:
                    question_text = item["text"]
                    question_domain = item["topic"]
                    temp = [question_id, question_text, question_domain]
                    results.append(temp)
                    
    
    return results

def writeout_csv(filepath, data, header):
    
    f = open(filepath, 'w', encoding='utf-8', newline="")
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
    
    f.close 
    
    
    

if __name__ == "__main__":
    
    data_dir = "./dataset/" # mention the work dir
    
    question_file_path = os.path.join(data_dir, "questions.en.jsonl")
    
    question_id_list = [15, 16, 17, 18, 19, 20, 35, 59, 60, 61, 62, 63, 64, 1449, 1452, 1453, 1493, 1495, 1496, 1497, 2715, 3224, 3391, 3427, 3428, 3429, 3430, 3431, 3468, 3469, 3470, 3471]
   
    results = get_question(question_file_path, question_id_list)
    
    question_output_file_path = "./question_selected.csv"
    
    writeout_csv(question_output_file_path, results, ["question_id", "text", "topic"])    
    
                    
                    
                
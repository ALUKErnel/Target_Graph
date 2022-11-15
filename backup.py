    # print(all_targets_list)
    # if target_num != 0:
    #     # for t in all_targets_list[:target_num]:
    #     #     train_dataset_target = XStanceDataset_target('train', None, train_file_path, t, tokenizer, 0, max_seq_len)
    #     #     # 确认是否需要这两个split
    #     #     # valid_dataset = XStanceDataset_target('valid', None, valid_file_path, t, tokenizer, 0, max_seq_len)
    #     #     # test_dataset = XStanceDataset_target('test', 'new_comments_defr', test_file_path, t, tokenizer, 0, max_seq_len)
    #     #     dataset_list.append(train_dataset_target)
            
    #     return dataset_list, all_targets_list[:target_num]
    # else:
    #     # for t in all_targets_list:
    #     #     train_dataset_target = XStanceDataset_target('train', None, train_file_path, t, tokenizer, 0, max_seq_len)
    #     #     # 确认是否需要这两个split
    #     #     # valid_dataset = XStanceDataset_target('valid', None, valid_file_path, t, tokenizer, 0, max_seq_len)
    #     #     # test_dataset = XStanceDataset_target('test', 'new_comments_defr', test_file_path, t, tokenizer, 0, max_seq_len)
    #     #     dataset_list.append(train_dataset_target)

    #     return dataset_list, all_targets_list
    
    
    
    
    # def get_datasets_main(train_file_path,
#                  valid_file_path,
#                  test_file_path,
#                  tokenizer,
#                  num_train_lines,
#                  max_seq_len,
#                  sorted_questions
#                  ):

#     train_dataset = XStanceDataset('train', None, train_file_path, tokenizer, num_train_lines, max_seq_len)
#     train_dataset.to_Y_t(sorted_questions)
#     valid_dataset = XStanceDataset('valid', None, valid_file_path, tokenizer, 0, max_seq_len)
#     # valid_dataset.to_Y_t(sorted_questions)
#     test_dataset = XStanceDataset('test', 'new_comments_defr', test_file_path, tokenizer, 0, max_seq_len)
#     # valid_dataset.to_Y_t(sorted_questions)


#     return train_dataset, valid_dataset, test_dataset


# def get_datasets(train_file_path,
#                  valid_file_path,
#                  test_file_path,
#                  tokenizer,
#                  num_train_lines,
#                  max_seq_len,
#                  ):

#     train_dataset = XStanceDataset('train', None, train_file_path, tokenizer, num_train_lines, max_seq_len)
#     valid_dataset = XStanceDataset('valid', None, valid_file_path, tokenizer, 0, max_seq_len)
#     test_dataset = XStanceDataset('test', 'new_comments_defr', test_file_path, tokenizer, 0, max_seq_len)


#     return train_dataset, valid_dataset, test_dataset
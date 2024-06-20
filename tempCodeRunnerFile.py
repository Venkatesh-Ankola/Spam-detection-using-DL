ham_msg = data[data.text_type =='ham']
spam_msg = data[data.text_type=='spam']
ham_msg=ham_msg.sample(n=len(spam_msg),random_state=42)
balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)
balanced_data.head()
balanced_data['label']=balanced_data['text_type'].map({'ham':0,'spam':1})
train_msg, test_msg, train_labels, test_labels =train_test_split(balanced_data['text'],balanced_data['label'],test_size=0.2,random_state=434)

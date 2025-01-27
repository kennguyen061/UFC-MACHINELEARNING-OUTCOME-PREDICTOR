#PREUFC FIGHTS DATA IS NOT USED
#DOESNT ACCOUNT FOR NEW DEBUTS
#DOESNT ACCOUNT FOR ATHLETE REGRESSION DUE TO AGE
#DOESNT ACCOUNT FOR INJURIES FROM PREVIOUS FIGHT
#DOESNT ACCOUNT FOR ATHLETES RETURNING AFTER A LONG LAY OFF
#TODO: FIX AVERAGE CALCULATION IF FIGHTER HAS NEVER FOUGHT OUT OF ONE CORNER

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#imports data from excel webscraped data
all_data = pd.read_csv("large_dataset.csv")
fighter_features = pd.read_csv("large_dataset.csv")

#Averages stat features between fighters in the red and blue corner
#knockdowns
avg_r_kd = fighter_features.groupby('r_fighter')['r_kd'].mean()
avg_b_kd = fighter_features.groupby('b_fighter')['b_kd'].mean()

#significant strikes landed
avg_r_sig_str = fighter_features.groupby('r_fighter')['r_sig_str'].mean()
avg_b_sig_str = fighter_features.groupby('b_fighter')['b_sig_str'].mean()

#significant strikes attempted
avg_r_sig_str_att = fighter_features.groupby('r_fighter')['r_sig_str_att'].mean()
avg_b_sig_str_att = fighter_features.groupby('b_fighter')['b_sig_str_att'].mean()

#significant strike accuracy
avg_r_sig_str_acc = fighter_features.groupby('r_fighter')['r_sig_str_acc'].mean()
avg_b_sig_str_acc = fighter_features.groupby('b_fighter')['b_sig_str_acc'].mean()

#strikes landed
avg_r_str = fighter_features.groupby('r_fighter')['r_str'].mean()
avg_b_str = fighter_features.groupby('b_fighter')['b_str'].mean()

#strikes attempted
avg_r_str_att = fighter_features.groupby('r_fighter')['r_str_att'].mean()
avg_b_str_att = fighter_features.groupby('b_fighter')['b_str_att'].mean()

#strike accuracy
avg_r_str_acc = fighter_features.groupby('r_fighter')['r_str_acc'].mean()
avg_b_str_acc = fighter_features.groupby('b_fighter')['b_str_acc'].mean()

#takedowns landed
avg_r_td = fighter_features.groupby('r_fighter')['r_td'].mean()
avg_b_td = fighter_features.groupby('b_fighter')['b_td'].mean()

#takedowns attempted
avg_r_td_att = fighter_features.groupby('r_fighter')['r_td_att'].mean()
avg_b_td_att = fighter_features.groupby('b_fighter')['b_td_att'].mean()

#takedown accuracy
avg_r_td_acc = fighter_features.groupby('r_fighter')['r_td_acc'].mean()
avg_b_td_acc = fighter_features.groupby('b_fighter')['b_td_acc'].mean()

#submission attempts
avg_r_sub_att = fighter_features.groupby('r_fighter')['r_sub_att'].mean()
avg_b_sub_att = fighter_features.groupby('b_fighter')['b_sub_att'].mean()

#ground control time in seconds
avg_r_ctrl_sec = fighter_features.groupby('r_fighter')['r_ctrl_sec'].mean()
avg_b_ctrl_sec = fighter_features.groupby('b_fighter')['b_ctrl_sec'].mean()

#strike defense percent
avg_r_str_def_total = fighter_features.groupby('r_fighter')['r_str_def_total'].mean()
avg_b_str_def_total = fighter_features.groupby('b_fighter')['b_str_def_total'].mean()

#takedown defense percent
avg_r_td_def_total = fighter_features.groupby('r_fighter')['r_td_def_total'].mean()
avg_b_td_def_total = fighter_features.groupby('b_fighter')['b_td_def_total'].mean()

#averages out stats
avg_kd = (avg_r_kd + avg_b_kd) / 2  
avg_sig_str = (avg_r_sig_str + avg_b_sig_str) / 2
avg_sig_str_att = (avg_r_sig_str_att + avg_b_sig_str_att) / 2
avg_sig_str_acc = (avg_r_sig_str_acc + avg_b_sig_str_acc) / 2
avg_str = (avg_r_str + avg_b_str) / 2
avg_str_att = (avg_r_sig_str_att + avg_b_sig_str_att) / 2
avg_str_acc = (avg_r_str_acc + avg_b_str_acc) / 2
avg_td = (avg_r_td + avg_b_td) / 2
avg_td_att = (avg_r_td_att + avg_b_td_att) / 2
avg_td_acc = (avg_r_td_acc + avg_b_td_acc) / 2
avg_sub_att = (avg_r_sub_att + avg_b_sub_att) / 2
avg_ctrl_sec = (avg_r_ctrl_sec + avg_b_ctrl_sec) / 2
avg_str_def_total = (avg_b_str_def_total + avg_r_str_def_total) / 2
avg_td_def_total = (avg_b_td_def_total + avg_r_td_def_total) / 2

#creates a panda dataframe with all the features
fighter_features = pd.DataFrame({
    'avg_kd': avg_kd,
    'avg_sig_str': avg_sig_str,
    'avg_sig_str_att': avg_sig_str_att,
    'avg_sig_str_acc': avg_sig_str_acc,
    'avg_str': avg_str,
    'avg_str_att': avg_str_att,
    'avg_str_acc': avg_str_acc,
    'avg_td': avg_td,
    'avg_td_att': avg_td_att,
    'avg_td_acc': avg_td_acc,
    'avg_sub_att': avg_sub_att,
    'avg_ctrl_sec': avg_ctrl_sec,
    'avg_str_def_total': avg_r_str_def_total,
    'avg_td_def_total': avg_td_def_total,
    #'win_rate': win_rate
})

fighter_features.reset_index(inplace=True)
fighter_features.rename(columns={'r_fighter': 'name'}, inplace=True)

#prints a preview of fighter_features
print(fighter_features.head())

#shows upcoming card differences vs each other in stats
upcoming_card = pd.read_csv("upcomingcard.csv")

#left outer joins the fighter features and upcoming fight tables
upcoming_fighter_1 = upcoming_card.merge(fighter_features, left_on='fighter1', right_on='index', how='left')
upcoming_fight_both = upcoming_fighter_1.merge(fighter_features, left_on='fighter2', right_on='index', suffixes=('_fighter1', '_fighter2'),how='left')

print(upcoming_fight_both.head())

#for every column of both fighters, calculates the differences from fighter 1 and fighter2
for col in ['avg_kd', 'avg_sig_str', 'avg_sig_str_att', 'avg_sig_str_acc', 
'avg_str', 'avg_str_att', 'avg_str_acc', 'avg_td', 'avg_td_att', 'avg_td_acc', 'avg_sub_att', 'avg_ctrl_sec', 'avg_str_def_total', 'avg_td_def_total']:
     upcoming_fight_both[f'Diff_{col}'] = upcoming_fight_both[f'{col}_fighter1'] - upcoming_fight_both[f'{col}_fighter2']


upcoming_fight_final = upcoming_fight_both[['fighter1','fighter2'] + [col for col in upcoming_fight_both.columns if 'Diff' in col]]

print(upcoming_fight_final.head())

#prepares training data and labels
training_encoded_r = all_data.merge(fighter_features, left_on='r_fighter', right_on='index', how='left')
training_encoded_both = training_encoded_r.merge(fighter_features, left_on='b_fighter', right_on='index',suffixes=('_fighter1', '_fighter2'),how='left')

for col in ['avg_kd', 'avg_sig_str', 'avg_sig_str_att', 'avg_sig_str_acc', 
'avg_str', 'avg_str_att', 'avg_str_acc', 'avg_td', 'avg_td_att', 'avg_td_acc', 'avg_sub_att', 'avg_ctrl_sec', 'avg_str_def_total', 'avg_td_def_total']:
     training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_fighter1'] - training_encoded_both[f'{col}_fighter2']

#trainining the data and renames red/blue to true/false for wins/losses
training_data = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]
training_labels = training_encoded_both['winner'].map({'Red': True, 'Blue': False})

print(training_data.head())
print(training_data.shape)
print(training_labels.head())

#Clean all NaN numbers for data missing from website
training_data_cleaned = training_data.dropna()
training_labels_cleaned = training_labels[training_data_cleaned.index]
print(training_data_cleaned.shape, training_labels_cleaned.shape)

upcoming_fight_final_cleaned = upcoming_fight_final.dropna()

#initiates logistical regression model with iterations of 2000
logreg = LogisticRegression(max_iter=2000)

#trains the cross validation scores with logisticregression with the training data and labels
cross_val_scores = cross_val_score(logreg, training_data_cleaned, training_labels_cleaned, cv=5)
cross_val_scores_mean = cross_val_scores.mean()
#prints the mean success rate of the model
print(cross_val_scores_mean)

# Train the logistic regression model on the entire cleaned training dataset
logreg.fit(training_data_cleaned, training_labels_cleaned)

#Predicts upcoming fights based off of upcomingcard.csv and prints it to Predictions.xlsx
#predicts fights based off of diff columns
upcoming_fight_prob = logreg.predict_proba(upcoming_fight_final_cleaned[[col for col in upcoming_fight_final_cleaned.columns if 'Diff_' in col]])
print(upcoming_fight_prob)

#Cleans up columns to print each fighter and rec corner's chance of winning
upcoming_r_win = upcoming_fight_prob[:, 1]
upcoming_fight_final_cleaned['Red Corner Win probability'] = upcoming_r_win
print("UPCOMING FIGHT\n")
print(upcoming_fight_final_cleaned.head())

upcoming_predictions = upcoming_fight_final_cleaned[['fighter1', 'fighter2', 'Red Corner Win probability']]

print(upcoming_predictions)
upcoming_predictions.to_excel("Predictions.xlsx")
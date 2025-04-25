import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# preprocessing
fw_data = pd.read_csv("raw_if_data.csv")
fw_data = fw_data.dropna()
fw_data.head()
fw_data['Action'] = fw_data.Action.astype('category')
features = ['Source Port', 'Destination Port', 'NAT Source Port',
            'Bytes', 'Bytes Sent', 'Bytes Received', 'Packets',
            'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']

# separate label from features
X = fw_data[features]
y = fw_data['Action']

# normalize features
X_norm = MinMaxScaler().fit_transform(X)

# chi-square test
chi_scores = chi2(X_norm,y)
scores_df = pd.DataFrame({'feature': features ,
                          'chi-square_vals': chi_scores[0],
                          'p_vals': chi_scores[1]})

print(scores_df.sort_values('chi-square_vals', ascending= False))

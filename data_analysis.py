# Rest of your imports
# Add this right after the imports
import matplotlib
matplotlib.use('Agg')  # Use Agg backend instead of Qt

# Rest of your imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Read the CSV file and select only the first 8 columns
df = pd.read_csv('data_1 sheet.csv', usecols=range(8))

# Clean the data - replace '#N/A' with pandas NA
df['Outcome'] = df['Outcome'].replace('#N/A', pd.NA)

# Create a deduplicated wins dataframe - one win per account
win_df = df[df['Outcome'] == 'Win']
unique_wins_df = win_df.drop_duplicates(subset=['Customer Account ID'])

# Standardize the 'Disposition' column
df['Disposition'] = df['Disposition'].replace('No interest', 'No Interest')

# 1. Win rate by pitch type
pitch_wins = df[df['Outcome'] == 'Win'].groupby('Pitch')['Customer Account ID'].count()
pitch_totals = df[df['Call Connect'] == 'Connected'].groupby('Pitch')['Customer Account ID'].count()
pitch_win_rates = (pitch_wins / pitch_totals * 100).round(2)

# 2. Initial engagement vs ultimate win rate
engagement_by_pitch = df[df['Call Connect'] == 'Connected'].groupby('Pitch')['Disposition'].value_counts()
engaged_to_win = df[df['Disposition'] == 'Engaged'].groupby('Pitch')['Outcome'].value_counts()

# 3. Disposition patterns by pitch
disposition_by_pitch = df[df['Call Connect'] == 'Connected'].groupby(['Pitch', 'Disposition']).size()
disposition_percentages = disposition_by_pitch.groupby(level=0).apply(lambda x: x / x.sum() * 100)

# 4. Conversion of "Engaged" to wins by pitch
engaged_outcomes = df[df['Disposition'] == 'Engaged'].groupby(['Pitch', 'Outcome']).size()
engaged_conversion_rates = engaged_outcomes.groupby(level=0).apply(lambda x: x / x.sum() * 100)

# 5. Detailed analysis of Engaged conversions
engaged_df = df[df['Disposition'] == 'Engaged']
engaged_outcomes = engaged_df.groupby(['Pitch', 'Outcome']).size().unstack(fill_value=0)
conversion_rates = (engaged_outcomes.div(engaged_outcomes.sum(axis=1), axis=0) * 100).round(2)
engaged_totals = engaged_df.groupby('Pitch').size()

# Print and write results
with open('analysis_summary.txt', 'w') as f:
    print("\nWin Rates by Pitch:")
    f.write("\nWin Rates by Pitch:\n")
    print(pitch_win_rates)
    f.write(pitch_win_rates.to_string() + "\n")

    print("\nDisposition Distribution by Pitch:")
    f.write("\nDisposition Distribution by Pitch:\n")
    print(disposition_percentages)
    f.write(disposition_percentages.to_string() + "\n")

    print("\nEngaged to Win Conversion by Pitch:")
    f.write("\nEngaged to Win Conversion by Pitch:\n")
    print(engaged_conversion_rates)
    f.write(engaged_conversion_rates.to_string() + "\n")

    print("\nTotal Engaged Conversations by Pitch:")
    f.write("\nTotal Engaged Conversations by Pitch:\n")
    print(engaged_totals)
    f.write(engaged_totals.to_string() + "\n")

    print("\nDetailed Conversion Rates from Engaged to Outcomes (%):")
    f.write("\nDetailed Conversion Rates from Engaged to Outcomes (%):\n")
    print(conversion_rates)
    f.write(conversion_rates.to_string() + "\n")

# Owner vs Manager analysis
stakeholder_responses = df[df['Call Connect'] == 'Connected'].groupby(['Person', 'Disposition']).size()
stakeholder_percentages = stakeholder_responses.groupby(level=0).apply(lambda x: x / x.sum() * 100)

engagement_rate = df[
    (df['Call Connect'] == 'Connected') & 
    (df['Disposition'] == 'Engaged')
].groupby('Person').size() / df[
    df['Call Connect'] == 'Connected'
].groupby('Person').size() * 100

print("\nStakeholder Response Patterns (%):")
print(stakeholder_percentages)

print("\nEngagement Rate by Stakeholder (%):")
print(engagement_rate)

# Add Owner Engagement Analysis
owner_df = df[df['Person'] == 'Owner']
owner_engaged_df = owner_df[owner_df['Disposition'] == 'Engaged']
owner_engaged_outcomes = owner_engaged_df['Outcome'].value_counts()
owner_engaged_percentages = (owner_engaged_outcomes / len(owner_engaged_df) * 100).round(2)

print("\n=== Owner Engagement Analysis ===")
print(f"\nTotal Owner engaged conversations: {len(owner_engaged_df)}")
print("\nOutcomes when Owner is engaged:")
print(owner_engaged_outcomes)
print("\nConversion rates when Owner is engaged:")
print(owner_engaged_percentages)

# Add Owner Engagement by Pitch Type Analysis
print("\n=== Owner Engagement Analysis by Pitch Type ===")
owner_engaged_df = df[(df['Person'] == 'Owner') & (df['Disposition'] == 'Engaged')]
engaged_by_pitch = owner_engaged_df.groupby('Pitch').size()
wins_by_pitch = owner_engaged_df[owner_engaged_df['Outcome'] == 'Win'].groupby('Pitch').size()
win_rate_by_pitch = (wins_by_pitch / engaged_by_pitch * 100).round(2)

print("\nTotal engaged Owner conversations by pitch type:")
print(engaged_by_pitch)
print("\nWin rate by pitch type when Owner is engaged:")
print(win_rate_by_pitch)

# Add Manager Deferral Analysis
print("\n=== Manager Deferral Analysis ===")
manager_deferrals_df = df[
    (df['Person'] == 'Manager') & 
    (df['Disposition'] == 'Deferred to Other Stakeholder')
]
total_deferrals = len(manager_deferrals_df)
deferral_outcomes = manager_deferrals_df['Outcome'].value_counts()
deferral_percentages = (deferral_outcomes / total_deferrals * 100).round(2)

print(f"\nTotal Manager deferrals: {total_deferrals}")
print("\nOutcomes of Manager deferrals:")
print(deferral_outcomes)
print("\nOutcome percentages for Manager deferrals:")
print(deferral_percentages)

# Add Funnel Analysis
print("\n=== Conversion Funnel Analysis ===")
total_attempts = len(df)
connected_calls = df[df['Call Connect'] == 'Connected']
total_connected = len(connected_calls)

funnel_stats = {
    'Total Attempts': total_attempts,
    'Connected': total_connected,
    'Connection Rate': f"{(total_connected/total_attempts * 100):.2f}%"
}

disposition_counts = connected_calls['Disposition'].value_counts()
for disposition in disposition_counts.index:
    count = disposition_counts[disposition]
    funnel_stats[f'{disposition}'] = count
    funnel_stats[f'{disposition} Rate (from Connected)'] = f"{(count/total_connected * 100):.2f}%"

for disposition in disposition_counts.index:
    disposition_df = connected_calls[connected_calls['Disposition'] == disposition]
    disposition_total = len(disposition_df)
    outcomes = disposition_df['Outcome'].value_counts()
    for outcome in outcomes.index:
        count = outcomes[outcome]
        funnel_stats[f'{disposition} to {outcome}'] = count
        funnel_stats[f'{disposition} to {outcome} Rate'] = f"{(count/disposition_total * 100):.2f}%"

print("\nDetailed Funnel Analysis:")
for metric, value in funnel_stats.items():
    print(f"{metric}: {value}")

# Calculate drop-off rates
print("\nDrop-off Analysis:")
connection_drop = total_attempts - total_connected
print(f"Failed Connections: {connection_drop} ({(connection_drop/total_attempts * 100):.2f}%)")

engaged = len(connected_calls[connected_calls['Disposition'] == 'Engaged'])
disposition_drop = total_connected - engaged
print(f"Lost at Disposition Stage: {disposition_drop} ({(disposition_drop/total_connected * 100):.2f}%)")

engaged_df = connected_calls[connected_calls['Disposition'] == 'Engaged']
wins = len(engaged_df[engaged_df['Outcome'] == 'Win'])
engaged_drop = engaged - wins
print(f"Lost at Final Stage: {engaged_drop} ({(engaged_drop/engaged * 100):.2f}%)")

# Update file output with funnel analysis
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Conversion Funnel Analysis ===\n")
    f.write("\nDetailed Funnel Analysis:\n")
    for metric, value in funnel_stats.items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nDrop-off Analysis:\n")
    f.write(f"Failed Connections: {connection_drop} ({(connection_drop/total_attempts * 100):.2f}%)\n")
    f.write(f"Lost at Disposition Stage: {disposition_drop} ({(disposition_drop/total_connected * 100):.2f}%)\n")
    f.write(f"Lost at Final Stage: {engaged_drop} ({(engaged_drop/engaged * 100):.2f}%)\n")

print("\n=== Multiple Touch Analysis ===")

# 1. Count accounts with multiple activities
account_touches = df['Customer Account ID'].value_counts()
multiple_touches = account_touches[account_touches > 1]

print(f"\nAccounts with multiple touches: {len(multiple_touches)}")
print("\nDistribution of number of touches per account:")
touch_count_distribution = account_touches.value_counts().sort_index()
print(touch_count_distribution)

# 2. Analyze disposition patterns for accounts with multiple touches
multi_touch_df = df[df['Customer Account ID'].isin(multiple_touches.index)]
multi_touch_df = multi_touch_df.sort_values(['Customer Account ID', 'Date'])

# Only look at accounts where we actually connected
connected_touches = multi_touch_df[multi_touch_df['Call Connect'] == 'Connected']
disposition_patterns = connected_touches.groupby('Customer Account ID')['Disposition'].agg(list)

print("\nDisposition Patterns Analysis:")
print(f"Accounts with multiple connected calls: {len(disposition_patterns)}")

# Count common disposition sequences
disposition_sequences = disposition_patterns.apply(lambda x: ' → '.join(x))
print("\nMost common disposition sequences:")
print(disposition_sequences.value_counts().head())

# 3. Analyze pitch changes and their outcomes
pitch_patterns = connected_touches.groupby('Customer Account ID')['Pitch'].agg(list)
accounts_with_pitch_changes = pitch_patterns[pitch_patterns.apply(lambda x: len(set(x)) > 1)]

print(f"\nAccounts with pitch changes: {len(accounts_with_pitch_changes)}")

# Analyze outcomes after pitch changes
final_outcomes = []
for account in accounts_with_pitch_changes.index:
    account_data = connected_touches[connected_touches['Customer Account ID'] == account]
    final_outcomes.append(account_data.iloc[-1]['Outcome'])

if final_outcomes:
    outcome_counts = pd.Series(final_outcomes).value_counts()
    print("\nOutcomes after changing pitch strategy:")
    print(outcome_counts)

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Multiple Touch Analysis ===\n")
    f.write(f"\nAccounts with multiple touches: {len(multiple_touches)}\n")
    f.write("\nDistribution of number of touches per account:\n")
    f.write(touch_count_distribution.to_string() + "\n")
    
    f.write("\nDisposition Patterns Analysis:\n")
    f.write(f"Accounts with multiple connected calls: {len(disposition_patterns)}\n")
    f.write("\nMost common disposition sequences:\n")
    f.write(disposition_sequences.value_counts().head().to_string() + "\n")
    
    f.write(f"\nAccounts with pitch changes: {len(accounts_with_pitch_changes)}\n")
    if final_outcomes:
        f.write("\nOutcomes after changing pitch strategy:\n")
        f.write(outcome_counts.to_string() + "\n")

print("\n=== Unique Winning Accounts Analysis ===")

# Get winning accounts (counting each account only once)
winning_df = df[df['Outcome'] == 'Win'].drop_duplicates(subset=['Customer Account ID'])
total_unique_wins = len(winning_df)

print(f"\nTotal Unique Winning Accounts: {total_unique_wins}")

# Breakdown by Person type
person_breakdown = winning_df['Person'].value_counts()
person_percentages = (person_breakdown / total_unique_wins * 100).round(2)
print("\nBreakdown by Person Type:")
print("Count:")
print(person_breakdown)
print("\nPercentages:")
print(person_percentages)

# Breakdown by Pitch type
pitch_breakdown = winning_df['Pitch'].value_counts()
pitch_percentages = (pitch_breakdown / total_unique_wins * 100).round(2)
print("\nBreakdown by Pitch Type:")
print("Count:")
print(pitch_breakdown)
print("\nPercentages:")
print(pitch_percentages)

# Combined Person-Pitch breakdown
combined_breakdown = winning_df.groupby(['Person', 'Pitch']).size()
combined_percentages = (combined_breakdown / total_unique_wins * 100).round(2)
print("\nDetailed Breakdown by Person and Pitch:")
print("Count:")
print(combined_breakdown)
print("\nPercentages:")
print(combined_percentages)

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Unique Winning Accounts Analysis ===\n")
    f.write(f"\nTotal Unique Winning Accounts: {total_unique_wins}\n")
    
    f.write("\nBreakdown by Person Type:\n")
    f.write("Count:\n")
    f.write(person_breakdown.to_string() + "\n")
    f.write("\nPercentages:\n")
    f.write(person_percentages.to_string() + "%\n")
    
    f.write("\nBreakdown by Pitch Type:\n")
    f.write("Count:\n")
    f.write(pitch_breakdown.to_string() + "\n")
    f.write("\nPercentages:\n")
    f.write(pitch_percentages.to_string() + "%\n")
    
    f.write("\nDetailed Breakdown by Person and Pitch:\n")
    f.write("Count:\n")
    f.write(combined_breakdown.to_string() + "\n")
    f.write("\nPercentages:\n")
    f.write(combined_percentages.to_string() + "%\n")

print("\n=== Success Rates by Combination Analysis (Eventual Wins) ===")

# First, identify accounts that eventually won
winning_accounts = df[df['Outcome'] == 'Win']['Customer Account ID'].unique()

# Add a column for eventual outcome
df['Eventually_Won'] = df['Customer Account ID'].isin(winning_accounts)

# Calculate success rates for all possible combinations based on eventual wins
success_rates = df.groupby(['Person', 'Disposition', 'Pitch'])['Eventually_Won'].apply(
    lambda x: x.mean() * 100  # Convert to percentage
).round(2)

# Add count of occurrences for each combination
combination_counts = df.groupby(['Person', 'Disposition', 'Pitch']).size()

# Combine rates and counts
combined_analysis = pd.DataFrame({
    'Success Rate (Eventual Win %)': success_rates,
    'Number of Attempts': combination_counts
}).sort_values('Success Rate (Eventual Win %)', ascending=False)

print("\nSuccess Rates for All Combinations (Based on Eventual Wins):")
print(combined_analysis)

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Success Rates by Combination Analysis (Eventual Wins) ===\n")
    f.write("\nSuccess Rates for All Combinations (Based on Eventual Wins):\n")
    f.write(combined_analysis.to_string() + "\n")

print("\n=== True Success Rate Analysis ===")

# For Managers
manager_attempts = df[df['Person'] == 'Manager'].groupby('Pitch')['Customer Account ID'].nunique()
manager_wins = df[
    (df['Person'] == 'Manager') & 
    (df['Outcome'] == 'Win')
].groupby('Pitch')['Customer Account ID'].nunique()
manager_success_rates = (manager_wins / manager_attempts * 100).round(2)

# For Owners
owner_attempts = df[df['Person'] == 'Owner'].groupby('Pitch')['Customer Account ID'].nunique()
owner_wins = df[
    (df['Person'] == 'Owner') & 
    (df['Outcome'] == 'Win')
].groupby('Pitch')['Customer Account ID'].nunique()
owner_success_rates = (owner_wins / owner_attempts * 100).round(2)

print("\nManager Success Rates (per 100 attempts):")
print("\nTotal Attempts by Pitch:")
print(manager_attempts)
print("\nTotal Wins by Pitch:")
print(manager_wins)
print("\nSuccess Rate (%):")
print(manager_success_rates)

print("\nOwner Success Rates (per 100 attempts):")
print("\nTotal Attempts by Pitch:")
print(owner_attempts)
print("\nTotal Wins by Pitch:")
print(owner_wins)
print("\nSuccess Rate (%):")
print(owner_success_rates)

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== True Success Rate Analysis ===\n")
    
    f.write("\nManager Success Rates (per 100 attempts):\n")
    f.write("\nTotal Attempts by Pitch:\n")
    f.write(manager_attempts.to_string() + "\n")
    f.write("\nTotal Wins by Pitch:\n")
    f.write(manager_wins.to_string() + "\n")
    f.write("\nSuccess Rate (%):\n")
    f.write(manager_success_rates.to_string() + "\n")
    
    f.write("\nOwner Success Rates (per 100 attempts):\n")
    f.write("\nTotal Attempts by Pitch:\n")
    f.write(owner_attempts.to_string() + "\n")
    f.write("\nTotal Wins by Pitch:\n")
    f.write(owner_wins.to_string() + "\n")
    f.write("\nSuccess Rate (%):\n")
    f.write(owner_success_rates.to_string() + "\n")

print("\n=== Feature Importance Analysis ===")

# Prepare data - only use rows where we actually connected
connected_df = df[df['Call Connect'] == 'Connected'].copy()

# Create dummy variables for categorical columns
X = pd.get_dummies(connected_df[['Person', 'Pitch', 'Disposition']])
y = (connected_df['Outcome'] == 'Win').astype(int)

# Create and fit decision tree
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10)
tree.fit(X, y)

# Get feature importance without visualization
importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance Scores (higher = more important):")
print(importance)

# Calculate success rates for each feature
print("\nSuccess Rates for Top Features:")
for feature in importance.head().index:
    # Get original column name and value
    col, val = feature.split('_', 1) if '_' in feature else (feature, None)
    
    if val:
        mask = connected_df[col] == val
        success_rate = (connected_df[mask]['Outcome'] == 'Win').mean() * 100
        print(f"\n{feature}:")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Total Occurrences: {mask.sum()}")

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Feature Importance Analysis ===\n")
    f.write("\nFeature Importance Scores (higher = more important):\n")
    f.write(importance.to_string() + "\n")

# Add after the Feature Importance Analysis section:

print("\n=== Visualizations ===")

# 1. Success Rates by Person and Pitch
plt.figure(figsize=(12, 6))
success_by_person_pitch = pd.DataFrame({
    'Manager': manager_success_rates,
    'Owner': owner_success_rates
})
success_by_person_pitch.plot(kind='bar')
plt.title('Success Rates by Person Type and Pitch')
plt.xlabel('Pitch Type')
plt.ylabel('Success Rate (%)')
plt.legend(title='Person Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('success_rates.png')
plt.close()

# 2. Feature Importance Visualization
plt.figure(figsize=(12, 6))
importance.head(10).plot(kind='bar')
plt.title('Top 10 Most Important Features')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 3. Conversion Funnel Visualization
plt.figure(figsize=(10, 6))
funnel_data = pd.Series({
    'Total Attempts': total_attempts,
    'Connected': total_connected,
    'Engaged': engaged,
    'Unique Wins': len(unique_wins_df)  # Changed from 'wins' to unique wins count
})
plt.bar(range(len(funnel_data)), funnel_data, color='skyblue')
plt.title('Conversion Funnel (Unique Wins)')
plt.xticks(range(len(funnel_data)), funnel_data.index, rotation=45)
plt.ylabel('Number of Calls')
for i, v in enumerate(funnel_data):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('conversion_funnel.png')
plt.close()

# 4. Disposition Distribution
plt.figure(figsize=(10, 6))
disposition_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of Dispositions (Connected Calls)')
plt.axis('equal')
plt.savefig('disposition_distribution.png')
plt.close()

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Visualization Files Generated ===\n")
    f.write("1. success_rates.png - Success rates by person type and pitch\n")
    f.write("2. feature_importance.png - Top 10 most important features\n")
    f.write("3. conversion_funnel.png - Overall conversion funnel\n")
    f.write("4. disposition_distribution.png - Distribution of dispositions\n")

print("\n=== Double Engagement Win Analysis ===")

# Get accounts that were called exactly twice
accounts_called_twice = account_touches[account_touches == 2].index

# Initialize counters
total_double_engaged = 0
double_engaged_wins = 0

# Analyze each account called twice
for account in accounts_called_twice:
    account_data = df[df['Customer Account ID'] == account].sort_values('Date')
    
    # Check if both calls resulted in engagement
    if len(account_data) == 2:  # Verify exactly two calls
        first_call = account_data.iloc[0]
        second_call = account_data.iloc[1]
        
        # Check if both calls were engaged
        if (first_call['Disposition'] == 'Engaged' and 
            second_call['Disposition'] == 'Engaged'):
            total_double_engaged += 1
            # Check if account won
            if any(account_data['Outcome'] == 'Win'):
                double_engaged_wins += 1

print(f"\nAccounts called exactly twice: {len(accounts_called_twice)}")
print(f"Accounts engaged on both calls: {total_double_engaged}")
print(f"Double-engaged accounts that won: {double_engaged_wins}")
if total_double_engaged > 0:
    win_rate = (double_engaged_wins / total_double_engaged) * 100
    print(f"Win rate for double-engaged accounts: {win_rate:.1f}%")

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Double Engagement Win Analysis ===\n")
    f.write(f"Accounts called exactly twice: {len(accounts_called_twice)}\n")
    f.write(f"Accounts engaged on both calls: {total_double_engaged}\n")
    f.write(f"Double-engaged accounts that won: {double_engaged_wins}\n")
    if total_double_engaged > 0:
        win_rate = (double_engaged_wins / total_double_engaged) * 100
        f.write(f"Win rate for double-engaged accounts: {win_rate:.1f}%\n")

print("\n=== Follow-up Analysis on Initially Engaged Accounts ===")

# Get accounts with multiple touches
accounts_with_followup = account_touches[account_touches > 1].index

# Initialize counters
initially_engaged = 0
initially_engaged_wins = 0

# Analyze each account with follow-ups
for account in accounts_with_followup:
    account_data = df[df['Customer Account ID'] == account].sort_values('Date')
    first_call = account_data.iloc[0]
    
    # Check if first call was engaged
    if first_call['Disposition'] == 'Engaged':
        initially_engaged += 1
        # Check if account eventually won
        if any(account_data['Outcome'] == 'Win'):
            initially_engaged_wins += 1

print(f"\nAccounts with follow-up calls: {len(accounts_with_followup)}")
print(f"Initially engaged accounts: {initially_engaged}")
print(f"Initially engaged accounts that won: {initially_engaged_wins}")
if initially_engaged > 0:
    win_rate = (initially_engaged_wins / initially_engaged) * 100
    print(f"Win rate for initially engaged accounts with follow-up: {win_rate:.1f}%")

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Follow-up Analysis on Initially Engaged Accounts ===\n")
    f.write(f"Accounts with follow-up calls: {len(accounts_with_followup)}\n")
    f.write(f"Initially engaged accounts: {initially_engaged}\n")
    f.write(f"Initially engaged accounts that won: {initially_engaged_wins}\n")
    if initially_engaged > 0:
        win_rate = (initially_engaged_wins / initially_engaged) * 100
        f.write(f"Win rate for initially engaged accounts with follow-up: {win_rate:.1f}%\n")

print("\n=== Analysis of Engaged but Lost Follow-ups ===")

# Initialize counters
initially_engaged_no_win = 0
total_followups_no_win = 0
disposition_after_engaged = []

# Analyze each account with follow-ups
for account in accounts_with_followup:
    account_data = df[df['Customer Account ID'] == account].sort_values('Date')
    first_call = account_data.iloc[0]
    
    # Check if first call was engaged
    if first_call['Disposition'] == 'Engaged':
        # Check if account did NOT win
        if not any(account_data['Outcome'] == 'Win'):
            initially_engaged_no_win += 1
            total_followups_no_win += len(account_data) - 1  # Subtract first call
            
            # Track dispositions of follow-up calls
            followup_dispositions = account_data.iloc[1:]['Disposition'].tolist()
            disposition_after_engaged.extend(followup_dispositions)

# Calculate average follow-ups per lost account
avg_followups = total_followups_no_win / initially_engaged_no_win if initially_engaged_no_win > 0 else 0

# Analyze follow-up dispositions
followup_disposition_counts = pd.Series(disposition_after_engaged).value_counts()
followup_disposition_percentages = (followup_disposition_counts / len(disposition_after_engaged) * 100).round(2)

print(f"\nInitially engaged accounts that didn't win: {initially_engaged_no_win}")
print(f"Average follow-up calls per lost account: {avg_followups:.1f}")
print("\nFollow-up call dispositions:")
for disposition, percentage in followup_disposition_percentages.items():
    print(f"  {disposition}: {percentage:.1f}%")

# Add to file output
with open('analysis_summary.txt', 'a') as f:
    f.write("\n=== Analysis of Engaged but Lost Follow-ups ===\n")
    f.write(f"Initially engaged accounts that didn't win: {initially_engaged_no_win}\n")
    f.write(f"Average follow-up calls per lost account: {avg_followups:.1f}\n")
    f.write("\nFollow-up call dispositions:\n")
    for disposition, percentage in followup_disposition_percentages.items():
        f.write(f"  {disposition}: {percentage:.1f}%\n")

import pandas as pd
import csv
file_path = "user_sessions_lastfm1k_minuser1000_minitem7_sessgap1200_minsesslen10_minhist50.csv"
output_file = "ordered_song_list.csv"

def get_ordered_song_list(file_path):
    cols = ['userid', 'timestamp', 'artid', 'artist_name', 'traid', 'track_name', 'session_id']
    
    df = pd.read_csv(
        file_path, 
        sep=',',  # <-- changed from \t
        header=None, 
        names=cols, 
        on_bad_lines='skip',
        quoting=csv.QUOTE_NONE,
        dtype=str
    )
    
    df.dropna(subset=['artist_name', 'track_name'], inplace=True)
    
    df = df[df['artist_name'].str.strip().astype(bool) & df['track_name'].str.strip().astype(bool)]
    
    df['artist_name'] = df['artist_name'].str.strip()
    df['track_name'] = df['track_name'].str.strip()
    
    df['artist_name'] = df['artist_name'].str.replace(',', ';', regex=False).str.replace('"', '', regex=False)
    df['track_name'] = df['track_name'].str.replace(',', ';', regex=False).str.replace('"', '', regex=False)
    
    freq_table = df.groupby(['artist_name', 'track_name']).size().reset_index(name='play_count')
    
    freq_table_sorted = freq_table.sort_values(by='play_count', ascending=False).reset_index(drop=True)
    
    return freq_table_sorted

ordered_songs_df = get_ordered_song_list(file_path)
ordered_songs_df.insert(0, 'track_index', range(len(ordered_songs_df)))
ordered_songs_df.to_csv(output_file, index=False)
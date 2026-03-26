import pandas as pd
import math
import string
import os
import copy
import sys

# Ballot importer function
def find_election_information(file: str, method: int = 0) -> tuple[str, int, int, pd.DataFrame, dict]:
    # Read file
    if file.endswith('.csv'):
        ballot_info = pd.read_csv(filepath_or_buffer=file, header=None, usecols=[0])
        lines = [str(l).strip() for l in ballot_info.iloc[:, 0].to_list() if str(l).strip()]
    else:  # assume it's a BLT file
        with open(file, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    # First row = candidates, winners
    num_candidates, num_winners = map(int, lines[0].split())
    ballot_lines = lines[1:lines.index("0")]
    title = lines[-1]

    # Candidate labels: 1 → A, 2 → B, etc.
    candidate_labels = {i+1: string.ascii_uppercase[i] for i in range(num_candidates)}

    # Parse ballots
    parsed_ballots = []
    for line in ballot_lines:
        parts = list(map(int, line.split()))
        count = parts[0]
        # 0 marks the end of a ballot
        ranks = "".join(candidate_labels[int(r)] for r in parts[1:] if int(r) != 0)
        parsed_ballots.append({"count": count, "ballot": ranks})

    if method == 4: # weight check method so i have to explode each individual ballot
        ballots_to_use = []
        for ballot in parsed_ballots:
            ballots_to_use.extend([{"count": 1, "ballot": ballot["ballot"]}] * ballot["count"])
    else:
        ballots_to_use = parsed_ballots
    # DataFrame
    ballot_df = pd.DataFrame(ballots_to_use, columns=["count", "ballot"])

    return title, num_candidates, num_winners, ballot_df, candidate_labels

# Example usage
#file = "Scotland data, LEAP party information/glasgow22/Ward2.blt"
#title, num_candidates, num_winners, ballot_dataframe, candidates = find_election_information(file)

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def Scottish_STV(frame, n, S, method:int = 0, raw_counts:bool = False, weight_threshold:float = 0) -> tuple[set, dict]:
    """frame is a dataframe with columns 'count' and 'ballot', n is number of candidates, S is number of seats, method is exhaustion method"""
    # Define lists
    winners=set()
    hopefuls = set(string.ascii_uppercase[0:n])

    # Turn count into float and ensure numeric types
    frame['count'] = pd.to_numeric(frame['count'], errors='coerce').fillna(0.0).astype(float)
    frame['original_count'] = frame['count'].copy()

    # Set up the original frame for exhaustion tracking
    frame['already_counted'] = False
    original_frame=frame.copy(deep=True)

    cand_dict={i: string.ascii_uppercase[i] for i in range(n)}
    quota=math.floor(sum(frame['count'])/(S+1))+1
    current_round = 1

    # Key exhaustion tracking statistics (EN's research focus)
    exhausted_by_round = {1.0: 0.0}  # round number: exhausted count


    def tabulate(frame:pd.DataFrame, n: int) -> tuple[dict, pd.DataFrame]:
        """Iterates over frame and counts first choice votes. Returns dictionary of vote counts and original frame."""
        vote_counts = {cand_dict[i]: 0.0 for i in range(n)}

        # mask valid (non-empty, non-null) ballots
        mask = frame['ballot'].notna() & (frame['ballot'] != '')

        if mask.any():
            # extract first preference letter for each valid ballot
            first_choices = frame.loc[mask, 'ballot'].str[0]
            # sum counts by first choice
            sums = frame.loc[mask].assign(first=first_choices).groupby('first', sort=False)['count'].sum()
            # update vote_counts dict with the grouped sums
            for letter, val in sums.items():
                if letter in vote_counts:
                    vote_counts[letter] = float(val) # type: ignore
                else:
                    # ignore unexpected labels (preserve existing behaviour)
                    continue

        return vote_counts, frame

    def remove_and_exhaust(current_round, frame:pd.DataFrame, cand, transfer_weight:float=1.0, elected:bool=False):
        """Removes candidate from all ballots in frame. If elected=True, also transfers votes with weight.
        Preserves original behavior: returns (frame, current_round, election_over) and updates exhausted_by_round.
        """
        election_over = False
        exhausted_this_round = float(0.0)
        current_round += 1

        # Collect indices to drop after we finish iterating
        indices_to_drop = []

        # For case 2/3 we only need to run the original_frame mask once per call.
        mask_case2_done = False
        mask_case3_done = False

        # iterate over actual index labels
        for k in list(frame.index):
            # read values once
            ballot = str(frame.at[k, 'ballot']) if 'ballot' in frame.columns else ''
            count_val = float(frame.at[k, 'count']) if 'count' in frame.columns else 0.0  # type: ignore

            # Remove candidate from ballot (if present)
            if cand and ballot != '':
                if elected and ballot[0] == cand:
                    # transfer the ballot's count by weight
                    frame.at[k, 'count'] = truncate(count_val * transfer_weight, 5)
                    # remove any already-elected winners from this ballot
                    b = ballot
                    for winner in list(winners):
                        if winner in b:
                            b = b.replace(winner, '')
                    frame.at[k, 'ballot'] = b
                    ballot = b  # update local copy
                elif cand in ballot:
                    new_b = ballot.replace(cand, '')
                    frame.at[k, 'ballot'] = new_b
                    ballot = new_b

            # Exhaustion checks per method
            match method:
                case 0:
                    # traditional/final-round style: if ballot empty, exhausted
                    if ballot == '':
                        if not raw_counts:
                            exhausted_this_round += float(frame.at[k, 'count'])  # type: ignore
                        else:
                            exhausted_this_round += float(frame.at[k, 'original_count'])  # type: ignore
                        indices_to_drop.append(k)

                case 1:
                    # same as case 0 for empty ballots
                    if ballot == '':
                        if not raw_counts:
                            exhausted_this_round += float(frame.at[k, 'count'])  # type: ignore
                        else:
                            exhausted_this_round += float(frame.at[k, 'original_count'])  # type: ignore
                        indices_to_drop.append(k)

                    # cutting final rounds: evaluate once (uses vote_counts from outer scope)
                    if not mask_case2_done:
                        sorted_counts = sorted(vote_counts.values(), reverse=True)
                        idx = S - len(winners) - 1
                        votes_to_beat = sorted_counts[idx] if 0 <= idx < len(sorted_counts) else 0
                        possible_sum = sum(vote for vote in vote_counts.values() if vote < votes_to_beat)
                        if possible_sum <= votes_to_beat:
                            election_over = True
                            leading_hopefuls = [k_ for k_, v in vote_counts.items() if v >= votes_to_beat and k_ in hopefuls]
                            winners.update(leading_hopefuls)
                            print("election over due to no one being able to catch up")
                            for h in leading_hopefuls:
                                hopefuls.discard(h)
                        mask_case2_done = True
                        if election_over:
                            break

                case 2:
                    # first-choice representation: run mask once per call
                    if not mask_case2_done:
                        mask = (
                            original_frame["ballot"].str[0].isin(winners)
                            & (~original_frame["already_counted"])
                            & original_frame["ballot"].notna()
                        )
                        exhausted_this_round += original_frame.loc[mask, "count"].sum()
                        original_frame.loc[mask, "already_counted"] = True
                        mask_case2_done = True

                case 3:
                    # any winner in top S: run mask once per call
                    if not mask_case3_done:
                        mask = (
                            original_frame["ballot"].notna()
                            & (~original_frame["already_counted"])
                            & original_frame["ballot"].apply(lambda b: bool(set(b[:n]) & winners))
                        )
                        exhausted_this_round += original_frame.loc[mask, "count"].sum()
                        original_frame.loc[mask, "already_counted"] = True
                        mask_case3_done = True

                case 4:
                    # weight-check exhaustion: mark ballots with low weight once
                    already = frame.at[k, 'already_counted'] if 'already_counted' in frame.columns else False
                    if float(frame.at[k, 'count']) <= weight_threshold and not already:  # type: ignore
                        exhausted_this_round += float(frame.at[k, 'original_count'])  # type: ignore[operator]
                        frame.at[k, 'already_counted'] = True
                        if ballot == '':
                            indices_to_drop.append(k)

                case _:
                    raise ValueError("Method not recognized")

        # Drop exhausted rows after iterating
        if indices_to_drop:
            frame.drop(index=indices_to_drop, inplace=True)

        # Record per-round exhausted (only when election not forced over by method logic)
        if not election_over:
            exhausted_this_round = truncate(exhausted_this_round, 5)
            exhausted_by_round[current_round] = exhausted_this_round

        return frame, current_round, election_over
    
    #print("quota ",quota)
    #print(frame)
    #print(cand_dict)
    #print(hopefuls)

    #Get each candidate's initial number of votes this round
    vote_counts, frame = tabulate(frame, n)

    for cand in hopefuls:
        if vote_counts[cand]==0:
            hopefuls.remove(cand)
    #print(vote_counts)
    #print('\n')
    max_count=max(vote_counts.values())
    while len(winners)<S:
        
        max_count=max(vote_counts.values())
        #somebody is elected and we have to transfer their votes
        if max_count>=quota:
            #There might be multiple people elected this round; save them as a sorted dictionary
            votes_for_winners={k:vote_counts[k] for k in vote_counts.keys() if vote_counts[k]>=quota}
            votes_for_winners=dict(sorted(votes_for_winners.items(),key=lambda x: x[1], reverse=True))

            # If our last winner exceeds quota
            #If we try to elect too many people, need to drop someone who surpassed quota
            if len(winners)+len(votes_for_winners)>=S:
                remaining_slots = S - len(winners)

                for cand in list(votes_for_winners.keys())[:remaining_slots]:
                    winners.add(cand)
                    hopefuls.discard(cand)

                    if len(winners) == S:
                        return winners, exhausted_by_round

            else:
                # Since we can elect all who surpassed quota, do it
                winners.update(votes_for_winners.keys())
                for cand in winners:
                    if cand in hopefuls:
                        hopefuls.remove(cand)
                #print(winners)

                while len(votes_for_winners)>0:
                    # Pull them off and add them to winners
                    cand=list(votes_for_winners.keys())[0]
                    
                    # Double-check
                    if cand not in winners:
                        winners.add(cand)
                        hopefuls.remove(cand)

                    if len(winners)==S:
                        return winners, exhausted_by_round
                    #print("cand elected", cand)
                    #print('surplus',vote_counts[cand]-quota)
                    
                    weight=truncate((vote_counts[cand]-quota)/vote_counts[cand],5)
                    frame, current_round, election_over = remove_and_exhaust(current_round, frame, cand, weight, elected=True)
                    if election_over: 
                        return winners, exhausted_by_round
                    #print('weight',weight)
                    votes_for_winners.pop(cand)
                    
                    # Retabulate
                    vote_counts, frame = tabulate(frame, n)

                    votes_for_winners={k:vote_counts[k] for k in vote_counts.keys() if vote_counts[k]>=quota }
                    votes_for_winners=dict(sorted(votes_for_winners.items(),key=lambda x: x[1], reverse=True))
                    #print(vote_counts)
                    for cand in votes_for_winners.keys():
                        if cand not in winners:
                            winners.add(cand)
                            hopefuls.remove(cand)
                    
                    if len(winners)==S:
                        
                        return winners, exhausted_by_round
                    if method != 4:
                        frame = frame.groupby("ballot", as_index=False).agg(count=("count", "sum"), original_count=("original_count", "sum"))
        #nobody is elected by surpassing quota, but the number
        #of candidates left equals S
        elif len(hopefuls)+len(winners)==S:
            print("election over due to hopefuls + winners = S")
            winners.update(hopefuls)
            hopefuls.clear()
            frame, current_round, election_over = remove_and_exhaust(current_round, frame, '')
            if exhausted_by_round[max(exhausted_by_round)] == 0:
                del exhausted_by_round[max(exhausted_by_round)]
            return winners, exhausted_by_round
        #remove weakest cand and transfer their votes with weight one
        else:
            #print(vote_counts)
            nonzero_votes = [i for i in vote_counts.values() if i > 0]
            if not nonzero_votes:
                #print("All remaining candidates have zero votes. Ending election.")
                winners.update(hopefuls)
                return winners, exhausted_by_round
            min_count = min(nonzero_votes)
            count=0
            for votes in vote_counts.values():
                if votes==min_count:
                    count+=1

            if count==1:
                eliminated_cand = str(list(vote_counts.keys())[list(vote_counts.values()).index(min_count)])
                #print("eliminate cand ",eliminated_cand)
                hopefuls.remove(eliminated_cand)
                frame, current_round, election_over = remove_and_exhaust(current_round, frame, eliminated_cand)
                if election_over:
                    return winners, exhausted_by_round
                #print(hopefuls)
                vote_counts, frame = tabulate(frame, n)

                #print(vote_counts)
                #print('\n')
                max_count=max(vote_counts.values())
                if method != 4: # speeds up when i don't need weights
                    frame = frame.groupby("ballot", as_index=False).agg(count=("count", "sum"), original_count=("original_count", "sum"))
                #print(frame)
            elif len(hopefuls)-count>=S-len(winners):
                eliminated_cands=[]
                for cand in vote_counts:
                    if vote_counts[cand]==min_count:
                        eliminated_cands.append(cand)
                #print('eliminated candidates', eliminated_cands)
                for cand in eliminated_cands:
                    #print("eliminate cand ", cand)
                    hopefuls.remove(cand)
                    frame, current_round, election_over = remove_and_exhaust(current_round, frame, cand)
                    if election_over:
                        return winners, exhausted_by_round
                    
                vote_counts, frame = tabulate(frame, n)
                #print(hopefuls)
                #print(vote_counts)
                #print('\n')
                max_count=max(vote_counts.values())
                if method != 4: # speeds up when i don't need weights
                    frame = frame.groupby("ballot", as_index=False).agg(count=("count", "sum"), original_count=("original_count", "sum"))
                
            else:
                #print('tie',count)
                #print(original_frame)
                return set(), exhausted_by_round
    return winners, exhausted_by_round

# Sample standalone usage
#title, num_candidates, num_winners, ballot_dataframe, candidates = find_election_information(file)
#wins, exhausted_by_round = Scottish_STV(ballot_dataframe,num_candidates,num_winners)

def export_exhausted_ballots(file, method = 0, weight_threshold: float = 0, raw_counts = False):
    _, num_candidates, num_winners, ballot_dataframe, _ = find_election_information(file, method)
    _, exhausted_by_round = Scottish_STV(ballot_dataframe.copy(deep = True),num_candidates,num_winners, method, raw_counts, weight_threshold)
    vote_total = int(ballot_dataframe['count'].sum())
    # Construct DataFrame from exhausted_by_round dictionary
    exhausted_df = pd.DataFrame(list(exhausted_by_round.items()), columns=['Round', 'Exhausted']) # type: ignore
    exhausted_df['Cumulative Exhausted'] = round(exhausted_df['Exhausted'].cumsum(),5)
    if (not raw_counts) or (method == 4):
        exhausted_df['% Exhausted'] = round(exhausted_df['Exhausted'] / vote_total * 100, 2)
        exhausted_df['% Cumulative'] = round(exhausted_df['% Exhausted'].cumsum(),2)
    else:
        exhausted_df['% Cumulative'] = round(exhausted_df['Cumulative Exhausted'] / vote_total, 2)
        exhausted_df['% Exhausted'] = round(exhausted_df["Exhausted"] / (vote_total - exhausted_df["Cumulative Exhausted"].shift(fill_value=0)) * 100, 2)
        pass
    exhausted_df.at[0,'key info'] = int(num_candidates)
    exhausted_df.at[1, 'key info'] = int(num_winners)
    exhausted_df.at[2, 'key info'] = vote_total
    exhausted_df.at[3, 'key info'] = int(method)
    if method == 4: # weight check
        exhausted_df.at[4, 'key info'] = weight_threshold # type: ignore
        method = str(method) + "_" + str(weight_threshold)
    else:
        exhausted_df.at[4, 'key info'] = 0
    
    # Get the base name of the input file (without extension)
    city_year = os.path.basename(os.path.dirname(file))
    base = os.path.splitext(os.path.basename(file))[0]

    exhausted_df.at[5, 'key info'] = int(base.lstrip("S"))

    # Create a folder path like results/{method}/{city_year}
    folder = os.path.join("processed_results", str(method), city_year) if not raw_counts else os.path.join("processed_results", "rawcount", str(method), city_year)
    os.makedirs(folder, exist_ok=True)

    # Build the filename
    csv_name = f"exhaust-{city_year}-{base}-{method}.csv"
    csv_path = os.path.join(folder, csv_name)

    # Save CSV
    exhausted_df.to_csv(csv_path, index=False)
    print(f"Saved {base}-{method}")

#file = "Scotland data, LEAP party information/glasgow22/Ward2.blt"
#export_exhausted_ballots(file, method=4, weight_threshold=0.05)

"""
# Iterate through main directory and subdirectories, run each method on each file
count_raw = False
directory = 'processed_files'
subdirs = [root for root, dirs, files in os.walk(directory)][1:]
for subdir in subdirs:
    for file in os.listdir(subdir):
        if file.endswith(".pdf"):  # skip PDFs
            continue
        filepath = os.path.join(subdir, file)

        # derive city_year + base exactly as export_exhausted_ballots does
        city_year = os.path.basename(os.path.dirname(filepath))
        base = os.path.splitext(os.path.basename(filepath))[0]
        i += 1
        if i > 5:
            raise TimeoutError("Stopping after 5 files for testing")
        print(filepath, f'{i}/785')
        for method in [0, 1, 2, 3]:
            if method == 4:
                for weight_threshold in [0.05, 0.01, 0.001, 0.0001]:
                    # exactly how the function does it
                    method_str = str(4) + "_" + str(weight_threshold)

                    folder = os.path.join("processed_ouputs", method_str, city_year) if not count_raw else os.path.join("processed_outputs", "rawcount", method_str, city_year)
                    csv_name = f"exhaust-{city_year}-{base}-{method_str}.csv"
                    csv_path = os.path.join(folder, csv_name)

                    if os.path.exists(csv_path):
                        print(f"Skipping {csv_path}, already exists")
                        continue

                    export_exhausted_ballots(filepath, 4, weight_threshold, count_raw)

            else:
                folder = os.path.join("processed_outputs", str(method), city_year) if not count_raw else os.path.join("processed_outputs", "rawcount", str(method), city_year)
                csv_name = f"exhaust-{city_year}-{base}-{method}.csv"
                csv_path = os.path.join(folder, csv_name)

                if os.path.exists(csv_path):
                    #print(f"Skipping {csv_path}, already exists")
                    continue

                export_exhausted_ballots(filepath, method, raw_counts=count_raw)
"""


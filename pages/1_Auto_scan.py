import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations, chain
import io
import zipfile
import ast
from itertools import product
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


st.set_page_config(page_title='Association rule mining app', layout='centered')

def ensure_quotes_in_csv(file_obj):
    # Read the contents of the uploaded file
    lines = file_obj.getvalue().decode("utf-8").splitlines()

    updated_lines = []
    for line in lines:
        line = line.strip()
        if not (line.startswith('"') and line.endswith('"')):
            line = f'"{line}"'
        updated_lines.append(line)
    
    # Convert the updated lines back to a file-like object
    updated_file_obj = io.StringIO("\n".join(updated_lines))
    return updated_file_obj

def load_transaction(file):
    updated_file = ensure_quotes_in_csv(file)
    data = pd.read_csv(updated_file)
    
    if data.shape[1] != 1:
        st.error("Uploaded CSV file must have exactly one column.")
        return None, None

    # Single column case
    transactions = data.iloc[:, 0].apply(lambda x: x.split(',')).tolist()
    
    unique_items = get_unique_item(transactions)
    return transactions, unique_items

def get_unique_item(transaction_list):
    all_items = [item for transaction in transaction_list for item in transaction[0].split(',')]
    unique_items = list(set(all_items))
    return unique_items

def count_occurences(itemsets, Transaction):
    count = 0
    for i in range(len(Transaction)):
        if set(itemsets).issubset(set(Transaction[i])):
            count += 1
    return count     

def get_frequent(itemsets, Transaction, min_support, prev_discarded):
    L = []
    supp_count = []
    new_discarded = []

    k = len(prev_discarded)

    for s in range(len(itemsets)):
        discarded_before = False
        if k > 0:
            for it in prev_discarded[k]:
                if set(it).issubset(set(itemsets[s])):
                    discarded_before = True
                    break
        if not discarded_before:
            count = count_occurences(itemsets[s], Transaction)
            if count/(len(Transaction)) >= min_support:
                L.append(itemsets[s])
                supp_count.append(count)
            else:
                new_discarded.append(itemsets[s])

    return L, supp_count, new_discarded

def join_two_itemsets(it1, it2, order):
    it1.sort(key=lambda x: order.index(x))
    it2.sort(key=lambda x: order.index(x))

    for i in range(len(it1)-1):
        if it1[i] != it2[i]:
            return []
        
    if order.index(it1[-1]) < order.index(it2[-1]):
        return it1 + [it2[-1]]
    
    return []

def join_set_itemsets(set_of_its, order):
    C = []
    for i in range(len(set_of_its)):
        for j in range(i+1, len(set_of_its)):
            it_out = join_two_itemsets(set_of_its[i], set_of_its[j], order)
            if len(it_out) > 0:
                C.append(it_out)
    return C

def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) +1)))

def write_rules(X, X_S, S, conf, supp,  num_transaction):
    return {
        'Rules': str(X),
        'Antecedent': str(S),
        'Consequent': str(X_S),
        'Confidence': f"{conf:.3f}",
        'Support': f"{(supp/num_transaction):.3f}"
    }

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col],format='%Y-%m-%d')
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def find_rule_matches(rules_df, transactions):
    rule_matches = []

    for _, rule in rules_df.iterrows():
        antecedent = ast.literal_eval(rule['Antecedent'])
        consequent = ast.literal_eval(rule['Consequent'])

        antecedent_set = set(antecedent)
        consequent_set = set(consequent)
        
        matching_indices = []
        for index, transaction in enumerate(transactions):
            transaction_set = set(transaction)
            if antecedent_set.issubset(transaction_set) and consequent_set.issubset(transaction_set):
                matching_indices.append(index+1)
        
        rule_matches.append((rule, matching_indices))

    return rule_matches


with st.sidebar:
    st.header('Inputs')
    uploaded_csv = st.file_uploader('Upload CSV file', type=['csv'], accept_multiple_files=False, key=None, help='The file must be in CSV format and only one column', on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

st.title("Association rule mining application")
main_container = st.container()

if uploaded_csv is not None:
    with main_container:
        Transaction, order = load_transaction(uploaded_csv)
        if Transaction is None:
            st.stop()
        st.subheader('Uploaded CSV file')
        display_dataframe = pd.read_csv(uploaded_csv, header=None)
        st.dataframe(data=display_dataframe, use_container_width=True)

        st.subheader('Minimum support and minimum confidence combinations')
        num_trans = len(Transaction)
        order = sorted(order)

        results = []
        rules_all_conf = []

        support_range = [x for x in np.arange(0.01, 1.0, 0.05)]
        confidence_range = [x for x in np.arange(0.01, 1.0, 0.05)]

        for min_support, min_conf_value in product(support_range, confidence_range):
            C = {}
            L = {}
            itemset_size = 1
            Discarded = {itemset_size : []}
            C.update({itemset_size : [[f] for f in order]})
            supp_count_L = {}
            f, supp, new_discarded = get_frequent(C[itemset_size], Transaction, min_support, Discarded)
            Discarded.update({itemset_size : new_discarded})
            L.update({itemset_size : f})
            supp_count_L.update({itemset_size : supp})

            while len(L[itemset_size]) > 0:
                itemset_size += 1
                C[itemset_size] = join_set_itemsets(L[itemset_size - 1], order)
                f, supp, new_discarded = get_frequent(C[itemset_size], Transaction, min_support, Discarded)
                Discarded.update({itemset_size: new_discarded})
                L.update({itemset_size: f})
                supp_count_L.update({itemset_size: supp})

            rules = []
            for k in range(2, itemset_size + 1):
                for itemset in L[k]:
                    subsets = powerset(itemset)
                    for s in subsets:
                        X_S = list(set(itemset) - set(s))
                        if len(X_S) > 0:
                            conf = count_occurences(itemset, Transaction) / count_occurences(s, Transaction)
                            if conf >= min_conf_value:
                                supp = count_occurences(itemset, Transaction)
                                lift = conf / (count_occurences(X_S, Transaction) / num_trans)
                                rule = write_rules(itemset, X_S, s, conf, supp, num_trans)
                                rules.append(rule)
            
            rules_df = pd.DataFrame(rules)
            if not rules_df.empty:
                frequent_itemsets = [
                    {
                        'Itemset': ', '.join(item),
                        'Support Count': supp_count / num_trans
                    }
                    for k in supp_count_L
                    for item, supp_count in zip(L[k], supp_count_L[k])
                ]
                frequent_itemsets_df = pd.DataFrame(frequent_itemsets)
                
                results.append({
                    'Support': f"{min_support:.3f}",
                    'Confidence': f"{min_conf_value:.3f}",
                    'Frequent Itemsets Count': len(frequent_itemsets),
                    'Association Rules Count': len(rules),
                    'Total rows': (len(frequent_itemsets) + len(rules)),
                })

        results_df = pd.DataFrame(results)
        resultsV2 = st.dataframe(data=results_df, selection_mode='single-row', on_select='rerun', use_container_width=True)
        selection = resultsV2.selection.rows

        if len(selection) > 0:
            min_support2 = float(results_df.iloc[selection]['Support'].iloc[0])
            min_conf_value2 = float(results_df.iloc[selection]['Confidence'].iloc[0])

            C = {}
            L = {}
            itemset_size = 1
            Discarded = {itemset_size : []}
            C.update({itemset_size : [[f] for f in order]})
            supp_count_L = {}
            f, supp, new_discarded = get_frequent(C[itemset_size], Transaction, min_support2, Discarded)
            Discarded.update({itemset_size : new_discarded})
            L.update({itemset_size : f})
            supp_count_L.update({itemset_size : supp})

            while len(L[itemset_size]) > 0:
                itemset_size += 1
                C[itemset_size] = join_set_itemsets(L[itemset_size - 1], order)
                f, supp, new_discarded = get_frequent(C[itemset_size], Transaction, min_support2, Discarded)
                Discarded.update({itemset_size: new_discarded})
                L.update({itemset_size: f})
                supp_count_L.update({itemset_size: supp})
            
            st.subheader('Frequent Itemsets')
            all_frequent_itemsets = []
            for k in range(1, itemset_size):
                for i, itemset in enumerate(L[k]):
                    all_frequent_itemsets.append({
                        'Itemset': ', '.join(itemset),
                        'Support Count': f"{(supp_count_L[k][i] / num_trans):.3f}"
                    })
            all_frequent_itemsets_df = pd.DataFrame(all_frequent_itemsets)
            st.dataframe(all_frequent_itemsets_df, use_container_width=True)


            st.header('Association rule')
            rules = []
            for k in range(2, itemset_size + 1):
                for itemset in L[k]:
                    subsets = powerset(itemset)
                    for s in subsets:
                        X_S = list(set(itemset) - set(s))
                        if len(X_S) > 0:
                            conf = count_occurences(itemset, Transaction) / count_occurences(s, Transaction)
                            if conf >= min_conf_value2:
                                supp = count_occurences(itemset, Transaction)
                                lift = conf / (count_occurences(X_S, Transaction) / num_trans)
                                rules.append(write_rules(itemset, X_S, s, conf, supp, num_trans))

            rules_df = pd.DataFrame(rules)
            filtered_rule = filter_dataframe(rules_df)
            st.dataframe(data=filtered_rule, use_container_width=True)
            rule_matches = find_rule_matches(filtered_rule, Transaction)
            expander = st.expander('Rows containing the rules')
            with expander:
                for rule, matches in rule_matches:
                    st.write(f"Rule: {rule['Antecedent']} -> {rule['Consequent']}")
                    st.write(f"Matching rows: {matches}")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf,'w') as csv_zip:
                csv_zip.writestr("Frequent_itemset.csv", all_frequent_itemsets_df.to_csv(index=False))
                csv_zip.writestr("Association_Rules.csv", rules_df.to_csv(index=False))
                csv_zip.writestr("Association_Rules_Filtered.csv", filtered_rule.to_csv(index=False))
            
            buf.seek(0)
            st.download_button(label="Download frequent item and association rule",data=buf.getvalue(),file_name="Frequent_itemset_and_rules.zip",mime="application/zip")
        else:
            st.warning('Choose a row in the table')

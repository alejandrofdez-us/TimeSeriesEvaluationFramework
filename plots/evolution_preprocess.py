import pandas as pd

def evolution_preprocess(ts1, ts2, header):

    generated_data_sample_df = pd.DataFrame(ts2, columns=header)
    args = {"seq_len" : len(ts1[:, 0]), "ori_data_sample" : ts1, "generated_data_sample" : ts2,
                "generated_data_sample_df" : generated_data_sample_df}

    return args
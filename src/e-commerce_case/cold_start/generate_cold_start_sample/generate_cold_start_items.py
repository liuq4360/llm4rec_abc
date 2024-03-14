from utils import get_df


def get_cold_start_items(path_review: str = "../../data/amazon_review/beauty/All_Beauty_5.json") -> set[str]:

    df_view = get_df(path_review)

    # 对unixReviewTime升序排序
    df_view.sort_values('unixReviewTime', ascending=True, inplace=True)
    df_view = df_view.reset_index(drop=True)

    rows_num = df_view.shape[0]
    train_num = int(rows_num * 0.7)

    train_df = df_view.head(train_num)
    test_df = df_view.iloc[train_num:]

    train_items = set(train_df['asin'].unique())  # 71个
    test_items = set(test_df['asin'].unique())  # 44个

    cold_start_items = test_items.difference(train_items)  # 14个

    return cold_start_items

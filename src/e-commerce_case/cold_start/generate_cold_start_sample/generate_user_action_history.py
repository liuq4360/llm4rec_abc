from utils import get_df


def get_user_history(path_review: str = "../../data/amazon_review/beauty/All_Beauty_5.json") -> dict:
    df_view = get_df(path_review)

    # 对unixReviewTime升序排序
    df_view.sort_values('unixReviewTime', ascending=True, inplace=True)
    df_view = df_view.reset_index(drop=True)

    rows_num = df_view.shape[0]
    train_num = int(rows_num * 0.7)

    train_df = df_view.head(train_num)

    grouped = train_df.groupby('reviewerID')
    """
        >>> grouped.get_group('A105A034ZG9EHO')
      overall  verified  reviewTime      reviewerID        asin              style reviewerName reviewText     summary  unixReviewTime vote image
1246      5.0      True  07 6, 2014  A105A034ZG9EHO  B0009RF9DW  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1247      5.0      True  07 6, 2014  A105A034ZG9EHO  B000FI4S1E                NaN      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1250      5.0      True  07 6, 2014  A105A034ZG9EHO  B0012Y0ZG2  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1252      5.0      True  07 6, 2014  A105A034ZG9EHO  B000URXP6E  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
1253      5.0      True  07 6, 2014  A105A034ZG9EHO  B0012Y0ZG2  {'Size:': ' 180'}      K. Mras        yum  Five Stars      1404604800  NaN   NaN
    """
    user_history_dict = {}
    for name, group in grouped:
        reviewerID = name
        asin = group['asin']
        user_history_dict[reviewerID] = set(asin)

    return user_history_dict

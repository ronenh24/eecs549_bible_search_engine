# Author: Ronen H

import os
import pandas as pd


class Relevance:
    def precision(self, ret_rel: list[int], cut_off: int = 15) -> float:
        '''
        Calculates the relevance precision at the cut off of the returned Bible chapters.

        ret_rel: 1 means returned Bible chapter was relevant. 0 otherwise.
        cut_off: Maximum number of returned Bible chapters to consider. Defaults to 15.

        Returns the relevance precision at the cut off of the returned Bible chapters.
        '''
        num_ret = min(len(ret_rel), cut_off)
        if num_ret > 0:
            return sum(ret_rel[:num_ret]) / num_ret
        else:
            return 0
    
    def evaluate_ranker_results(self, test_queries_path: str, ranker, cut_off: int = 15) -> list[tuple[str, float]]:
        '''
        Evaluates how well the ranker does on the test set.

        test_queries_path: Path to test queries relevances csv.
        ranker: Ranker to test.
        cut_off: Maximum number of returned Bible chapters to consider. Defaults to 15.

        Returns precision at the cut off of each test query.
        '''
        precision_scores = []

        if not os.path.isfile(test_queries_path):
            raise Exception('Test Queries Relevances file does not exist.')
        relevance_df = pd.read_csv(test_queries_path)

        for query in relevance_df['query'].unique():
            print(query)
            rel_chapters = relevance_df[relevance_df['query'] == query][['chapterid', 'relevance']].sort_values('relevance', ascending=False)
            all_ret_chapters = ranker.query(query)
            ret_chapter_precision_labels = []

            # Relevance ratings out of 5. At least 4 for relevance.
            for ret_chapter in all_ret_chapters:
                ret_chapter_df = rel_chapters[rel_chapters['chapterid'] == ret_chapter[0]]
                if ret_chapter_df.empty:
                    ret_chapter_precision_labels.append(0)
                else:
                    rel = ret_chapter_df['relevance'].iloc[0]
                    if rel < 4:
                        ret_chapter_precision_labels.append(0)
                    else:
                        ret_chapter_precision_labels.append(1)

            query_precision = self.precision(ret_chapter_precision_labels, cut_off)
            print(query_precision)
            print()
            precision_scores.append((query, query_precision))

        return precision_scores


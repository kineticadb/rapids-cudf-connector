import math
import uuid
import time
import gpudb
import pandas as pd
import cudf


class CuDFMix(object):

    def to_df(self, sql):

        chunk_len = 2000
        ptbl = '_'.join(['kml_ptbl',
                         str(uuid.uuid4().hex[0:8]),
                         str(math.floor(time.time()))])

        ret_obj = self.execute_sql_and_decode(sql, offset=0, limit=chunk_len,
                                              options={'paging_table': ptbl})

        if not ret_obj['has_more_records']:
            return pd.DataFrame.from_dict(ret_obj['records'])

        else:
            pd_arr = []
            pd_arr.append(pd.DataFrame.from_dict(ret_obj['records']))

            total_record_count = ret_obj['total_number_of_records']
            remaining_count = total_record_count - chunk_len
            pages = math.ceil(remaining_count / chunk_len)
            page = 1

            while page <= pages:
                ret_obj = self.execute_sql_and_decode(sql,
                                                      offset=page * chunk_len,
                                                      limit=chunk_len,
                                                      options={'paging_table': ptbl})

                pd_arr.append(pd.DataFrame.from_dict(ret_obj['records']))
                page = page + 1

            self.clear_table(ptbl)
            return pd.concat(pd_arr)

    def to_cudf(self, sql):

        pdf = self.to_df(sql)
        return cudf.from_pandas(pdf)


class GPUdb(gpudb.GPUdb, CuDFMix):
    pass

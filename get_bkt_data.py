import pandas as pd
import numpy as np
import sqlite3

if __name__ == '__main__':

    conn = sqlite3.connect("pyreness.db")
    c = conn.cursor()

    QLG = {}
    POST = {}
    sql2 = "SELECT * from pyreness_scores;"
    data_y = pd.read_sql(sql2, conn)

    for each in data_y.loc[:, ['userID', 'QNLG', 'post']].values:
        QLG[each[0]] = each[1]
        POST[each[0]] = each[2]

    sql3 = "select distinct sid from pyreness;"
    student = pd.read_sql(sql3, conn)
    print "number of students: ", len(student)

    for i in xrange(10):        # for 10 KCs

        print " <<<<<<<<< kc " + str(i)
        col = 'kc'+str(i)
        tmp_col = []

        bkt_tutor = []
        bkt_perftime = []
        y1, y2 = [], []
        num = 0

        for key in student.values:
            tmp_sql = "SELECT performance_time FROM pyreness WHERE kcID = '" + str(i) + "' and " \
                      "sid = '" + str(key[0]) + "'; "
            res = c.execute(tmp_sql).fetchall()
            pt = []
            for each in res:
                pt.append(str(each[0]))
            if res:
                num += 1
                tmp_col.append(1)
            else:
                tmp_col.append(0)

            bkt_perftime.append(pt)
    
            y1.append(QLG[key[0]])
            y2.append(POST[key[0]])

        student[col] = tmp_col
        print len(bkt_perftime), len(y1), num
        np.save("bkt/new_bkt_pertime_kc" + str(i), np.array(bkt_perftime))
        np.save("bkt/new_bkt_tutor_kc" + str(i), np.array(bkt_tutor))

        np.save("bkt/pyrenees_qlg", np.array(y1))
        np.save("bkt/pyrenees_post", np.array(y2))

    student['QLG'] = y1
    student['POST'] = y2
    print "End"

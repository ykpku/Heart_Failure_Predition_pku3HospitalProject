

def __write_search_results(ful_filename, session_names, args_results, pred_results, wm='w'):
    with open(ful_filename, wm, encoding='utf-8') as f:
        for i in range(len(session_names)):
            f.write("-------------" + session_names[i] + "-------------\n")
            for ak in args_results[i].keys():
                f.write(ak + " : " + str(args_results[i][ak]) + "\t")
            f.write("\n")
            for rk in pred_results[i].keys():
                f.write(rk + " : " + str(pred_results[i][rk]) + "\t")
            f.write("\n")

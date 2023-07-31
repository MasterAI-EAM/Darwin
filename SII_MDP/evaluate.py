import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def test_classification():
    # pce = []
    # pce_hat = []
    pce_true = []
    pce_pred = []
    label_list = ["HIGH", "LOW", "NORMAL"]

    f = open("C:\\Users\\94833\\Desktop\\classification40_output.txt", encoding="utf-8")
    lines = f.readlines()
    line_index = 0
    output_list = []
    answer_list = []

    output_text = ''
    answer_text = ''
    for line in lines:
        if line_index % 24 == 19:
            output_text = str(line)
        elif line_index % 24 == 20 or line_index % 24 == 21:
            output_text = output_text + str(line)
        elif line_index % 24 == 22:
            output_text = output_text + str(line).split("@")[0]
            output_list.append(output_text)
        elif line_index % 24 == 23:
            answer_text = str(line).split("->")[1].split("@")[0].replace(",", "\n")
            answer_list.append(answer_text)
        line_index = line_index + 1

    print(output_list)
    print(answer_list)

    label_correct_num = 0
    for i in range(len(answer_list)):
        print("answer list len: " + str(len(answer_list)))
        answers = answer_list[i].split("\n")
        results = output_list[i].split("\n")
        if len(answers) != len(results):
            print("schema(s) missing")
            continue
        for schema_i in range(len(results)):
            answer_schema = answers[schema_i].split(":")[0].strip()
            result_schema = results[schema_i].split(":")[0].strip()
            """
            if answer_schema != result_schema:
                print("schema unaligned:" + answer_schema + " <> " + result_schema)
                continue
            """
            schema_answer = answers[schema_i].split(":")[1].strip()
            schema_result = results[schema_i].split(":")[1].strip()

            if "PCE" in result_schema:
                # pce.append(answer_figure)
                # pce_hat.append(result_figure)
                pce_true.append(schema_answer)
                pce_pred.append(schema_result)
                if schema_answer == schema_result:
                    label_correct_num = label_correct_num + 1

        print("______________________________")

    print("label_correct_ratio: " + str(label_correct_num/len(answer_list)*100) + "%")

    # mae_pce = np.mean(np.abs(np.array(pce)-np.array(pce_hat)))
    # print("mae_pce: " + str(mae_pce))
    # rmse_pce = np.sqrt(np.mean(np.square(np.array(pce) - np.array(pce_hat))))
    # print("rmse_pce: " + str(rmse_pce))
    # mean_pce = np.mean(pce)
    # print("mean_pce: " + str(mean_pce))
    # var_pce = np.var(pce)
    # print("var_pce: " + str(var_pce))
    print("______________________________")

    matrix = confusion_matrix(pce_true, pce_pred, labels=label_list)
    res = classification_report(pce_true, pce_pred, target_names=label_list)
    print(matrix, "\n")
    print(res)


def test_regression():
    voc = []
    voc_hat = []
    jsc = []
    jsc_hat = []
    pce = []
    pce_hat = []

    f = open("C:\\Users\\94833\\Desktop\\regression40_382_90output.txt", encoding="utf-8")
    lines = f.readlines()
    line_index = 0
    output_list = []
    answer_list = []

    output_text = ''
    answer_text = ''
    for line in lines:
        if line_index % 27 == 19:
            output_text = str(line)
        elif line_index % 27 == 20:
            output_text = output_text + str(line)
        elif line_index % 27 == 22:
            output_text = output_text + str(line).split("@")[0]
            output_list.append(output_text)
        elif line_index % 27 == 23:
            answer_text = str(line).split("->")[1]
        elif line_index % 27 == 24:
            answer_text = answer_text + str(line)
        elif line_index % 27 == 26:
            answer_text = answer_text + str(line).split("@")[0]
            answer_list.append(answer_text)
        line_index = line_index + 1

    print(output_list)
    print(answer_list)

    for i in range(len(answer_list)):
        print("answer list len: " + str(len(answer_list)))
        print("output list len: " + str(len(output_list)))
        answers = answer_list[i].split("\n")
        results = output_list[i].split("\n")
        if len(answers) != len(results):
            print("schema(s) missing")
            continue
        for schema_i in range(len(results)):
            answer_schema = answers[schema_i].split(":")[0].strip()
            result_schema = results[schema_i].split(":")[0].strip()
            """
            if answer_schema != result_schema:
                print("schema unaligned:" + answer_schema + " <> " + result_schema)
                continue
            """

            schema_answer = answers[schema_i].split(":")[-1].strip()
            schema_result = results[schema_i].split(":")[-1].strip()

            answer_figure = float(schema_answer)
            result_figure = float(schema_result)

            # result_schema == "JV default Voc":
            if "Voc" in result_schema:
                voc.append(answer_figure)
                voc_hat.append(result_figure)

            # result_schema == "JV default Jsc":
            elif "Jsc" in result_schema:
                jsc.append(answer_figure)
                jsc_hat.append(result_figure)

            # result_schema == "JV default PCE":
            elif "PCE" in result_schema:
                pce.append(answer_figure)
                pce_hat.append(result_figure)
                print("answer: " + str(answer_figure))
                print("result: " + str(result_figure))

        print("______________________________")

    mae_voc = np.mean(np.abs(np.array(voc) - np.array(voc_hat)))
    mae_jsc = np.mean(np.abs(np.array(jsc) - np.array(jsc_hat)))
    mae_pce = np.mean(np.abs(np.array(pce)-np.array(pce_hat)))
    print("mae_voc: " + str(mae_voc))
    print("mae_jsc: " + str(mae_jsc))
    print("mae_pce: " + str(mae_pce))
    print("______________________________")

    rmse_voc = np.sqrt(np.mean(np.square(np.array(voc) - np.array(voc_hat))))
    rmse_jsc = np.sqrt(np.mean(np.square(np.array(jsc) - np.array(jsc_hat))))
    rmse_pce = np.sqrt(np.mean(np.square(np.array(pce) - np.array(pce_hat))))
    print("rmse_voc: " + str(rmse_voc))
    print("rmse_jsc: " + str(rmse_jsc))
    print("rmse_pce: " + str(rmse_pce))
    print("______________________________")

    mean_voc = np.mean(voc)
    mean_jsc = np.mean(jsc)
    mean_pce = np.mean(pce)
    print("mean_voc: " + str(mean_voc))
    print("mean_jsc: " + str(mean_jsc))
    print("mean_pce: " + str(mean_pce))
    print("______________________________")

    var_voc = np.var(voc)
    var_jsc = np.var(jsc)
    var_pce = np.var(pce)
    print("var_voc: " + str(var_voc))
    print("var_jsc: " + str(var_jsc))
    print("var_pce: " + str(var_pce))
    print("______________________________")


def draw_regression():
    voc = []
    voc_hat = []
    jsc = []
    jsc_hat = []
    pce = []
    pce_hat = []

    f = open("C:\\Users\\94833\\Desktop\\regression40_382_output.txt", encoding="utf-8")
    lines = f.readlines()
    line_index = 0
    output_list = []
    answer_list = []

    output_text = ''
    answer_text = ''
    for line in lines:
        if line_index % 26 == 18:
            output_text = str(line).split("###")[1]
        elif line_index % 26 == 19:
            output_text = output_text + str(line)
        elif line_index % 26 == 21:
            output_text = output_text + str(line).split("@")[0]
            output_list.append(output_text)
        elif line_index % 26 == 22:
            answer_text = str(line).split("->")[1]
        elif line_index % 26 == 23:
            answer_text = answer_text + str(line)
        elif line_index % 26 == 25:
            answer_text = answer_text + str(line).split("@")[0]
            answer_list.append(answer_text)
        line_index = line_index + 1

    print(output_list)
    print(answer_list)

    for i in range(len(answer_list)):
        answers = answer_list[i].split("\n")
        results = output_list[i].split("\n")
        if len(answers) != len(results):
            print("schema(s) missing")
            continue
        for schema_i in range(len(results)):
            answer_schema = answers[schema_i].split(":")[0].strip()
            result_schema = results[schema_i].split(":")[0].strip()
            """
            if answer_schema != result_schema:
                print("schema unaligned:" + answer_schema + " <> " + result_schema)
                continue
            """

            schema_answer = answers[schema_i].split(":")[1].strip()
            schema_result = results[schema_i].split(":")[1].strip()

            answer_figure = float(schema_answer)
            result_figure = float(schema_result)

            # result_schema == "JV default Voc":
            if "Voc" in result_schema:
                voc.append(answer_figure)
                voc_hat.append(result_figure)

            # result_schema == "JV default Jsc":
            elif "Jsc" in result_schema:
                jsc.append(answer_figure)
                jsc_hat.append(result_figure)

            # result_schema == "JV default PCE":
            elif "PCE" in result_schema:
                pce.append(answer_figure)
                pce_hat.append(result_figure)
                print("answer: " + str(answer_figure))
                print("result: " + str(result_figure))

        print("______________________________")

    plt.figure(figsize=(10.5, 3.5))
    plt.subplot(131)
    x_voc = np.array(voc)
    y_voc = np.array(voc_hat)
    plt.xlim(0.5, 1.2)
    plt.ylim(0.5, 1.2)
    x = np.linspace(0.5, 1.2, 50)
    y = x
    plt.plot(x_voc, y_voc, 'o')
    plt.plot(x, y, color='green', linewidth=1.0, linestyle='--')
    plt.xlabel("Experimental Voc (V)")
    plt.ylabel("Predicted Voc (V)")

    plt.subplot(132)
    x_jsc = np.array(jsc)
    y_jsc = np.array(jsc_hat)
    plt.xlim(5, 25)
    plt.ylim(5, 25)
    x = np.linspace(5, 25, 50)
    y = x
    plt.plot(x_jsc, y_jsc, 'o')
    plt.plot(x, y, color='green', linewidth=1.0, linestyle='--')
    plt.xlabel("Experimental Jsc (mA/cm$^2$)")
    plt.ylabel("Predicted Jsc (mA/cm$^2$)")

    plt.subplot(133)
    x_pce = np.array(pce)
    y_pce = np.array(pce_hat)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    x = np.linspace(0, 20, 50)
    y = x
    plt.plot(x_pce, y_pce, 'o')
    plt.plot(x, y, color='green', linewidth=1.0, linestyle='--')
    plt.xlabel("Experimental PCE (%)")
    plt.ylabel("Predicted PCE (%)")

    plt.tight_layout()
    plt.savefig("./regression.png")
    plt.show()



#test_classification()
test_regression()
#draw_regression()

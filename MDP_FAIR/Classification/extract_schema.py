import json
import csv
import openai
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


openai.api_key = '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'
# ft_model = "davinci:ft-greendynamics-2023-03-29-17-28-16"
# ft_model = "davinci:ft-greendynamics-2023-03-30-06-27-48" # regression
ft_model = "davinci:ft-greendynamics-2023-03-30-16-41-28"  # classification


def extract_schema():
    with open("finetune_all.csv", "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        header = ['prompt', 'completion']
        writer.writerow(header)

    with open(r"C:\Users\94833\Desktop\store_gpt.json", "r", encoding="utf-8") as f:
        for dict_item in json.load(f):
            prompt = \
                "Substrate stack sequence: " + dict_item["Substrate_stack_sequence"] + \
                ", ETL stack sequence: " + dict_item["ETL_stack_sequence"] + \
                ", ETL additives compounds: " + dict_item["ETL_additives_compounds"] + \
                ", ETL deposition procedure: " + dict_item["ETL_deposition_procedure"] + \
                ", Perovskite composition long form: " + dict_item["Perovskite_composition_long_form"] + \
                ", Perovskite composition short form: " + dict_item["Perovskite_composition_short_form"] + \
                ", Perovskite additives compounds: " + dict_item["Perovskite_additives_compounds"] + \
                ", Perovskite deposition solvents: " + dict_item["Perovskite_deposition_solvents"] + \
                ", Perovskite deposition procedure: " + dict_item["Perovskite_deposition_procedure"] + \
                ", Perovskite deposition thermal annealing temperature: " \
                + dict_item["Perovskite_deposition_thermal_annealing_temperature"] + \
                ", Perovskite deposition thermal annealing time: " \
                + dict_item["Perovskite_deposition_thermal_annealing_time"] + \
                ", HTL stack sequence: " + dict_item["HTL_stack_sequence"] + \
                ", HTL additives compounds: " + dict_item["HTL_additives_compounds"] + \
                ", HTL deposition procedure: " + dict_item["HTL_deposition_procedure"] + \
                ", Backcontact stack sequence: " + dict_item["Backcontact_stack_sequence"] + \
                ", Backcontact additives compounds: " + dict_item["Backcontact_additives_compounds"] + \
                ", Backcontact deposition procedure: " + dict_item["Backcontact_deposition_procedure"] + \
                ", Stability measured: " + dict_item["Stability_measured"] + \
                ", Stability average over n number of cells: " + dict_item["Stability_average_over_n_number_of_cells"] + \
                ", Stability temperature range: " + dict_item["Stability_temperature_range"] + \
                ", Stability atmosphere: " + dict_item["Stability_atmosphere"] + \
                ", Stability time total exposure: " + dict_item["Stability_time_total_exposure"] + \
                ", Stability PCE initial value: " + dict_item["Stability_PCE_initial_value"] + \
                ", Stability PCE end of experiment: " + dict_item["Stability_PCE_end_of_experiment"] + \
                ", Cell area measured: " + dict_item["Cell_area_measured"] + \
                ", Cell number of cells per substrate: " + dict_item["Cell_number_of_cells_per_substrate"] + \
                ", Cell architecture: " + dict_item["Cell_architecture"] + \
                ", Cell flexible: " + dict_item["Cell_flexible"] + \
                ", Cell semitransparent: " + dict_item["Cell_semitransparent"] + \
                ", Cell semitransparent wavelength range: " + dict_item["Cell_semitransparent_wavelength_range"] + \
                ", Module: " + dict_item["Module"] + \
                ", JV average over n number of cells: " + dict_item["JV_average_over_n_number_of_cells"] + \
                ", JV light intensity: " + dict_item["JV_light_intensity"] + \
                ", JV light spectra: " + dict_item["JV_light_spectra"]

            completion = \
                "JV default Voc: " + dict_item["JV_default_Voc"] + \
                ", JV default Jsc: " + dict_item["JV_default_Jsc"] + \
                ", JV default FF: " + dict_item["JV_default_FF"] + \
                ", JV default PCE: " + dict_item["JV_default_PCE"]

            line = [prompt, completion]
            with open("finetune_all.csv", "a", encoding="utf-8", newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(line)


def extract_schema2():
    with open("finetune_classification_EXLOW11.csv", "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        header = ['prompt', 'completion']
        writer.writerow(header)
    with open("finetune_classification_LOW11.csv", "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        header = ['prompt', 'completion']
        writer.writerow(header)
    with open("finetune_classification_NORMAL11.csv", "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        header = ['prompt', 'completion']
        writer.writerow(header)
    with open("finetune_classification_HIGH11.csv", "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        header = ['prompt', 'completion']
        writer.writerow(header)
    with open("finetune_classification_TOP11.csv", "w", encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        header = ['prompt', 'completion']
        writer.writerow(header)

    with open(r"C:\Users\94833\Desktop\store_gpt.json", "r", encoding="utf-8") as f:
        for dict_item in json.load(f):
            prompt = \
                "Substrate stack sequence: " + dict_item["Substrate_stack_sequence"] + \
                ", ETL stack sequence: " + dict_item["ETL_stack_sequence"] + \
                ", ETL additives compounds: " + dict_item["ETL_additives_compounds"] + \
                ", ETL deposition procedure: " + dict_item["ETL_deposition_procedure"] + \
                ", ETL thickness: " + dict_item["ETL_thickness"] + \
                ", Perovskite dimension list of layers: " + dict_item["Perovskite_dimension_list_of_layers"] + \
                ", Perovskite composition a ions: " + dict_item["Perovskite_composition_a_ions"] + \
                ", Perovskite composition a ions coefficients: " \
                + dict_item["Perovskite_composition_a_ions_coefficients"] + \
                ", Perovskite composition b ions: " + dict_item["Perovskite_composition_b_ions"] + \
                ", Perovskite composition b ions coefficients: " \
                + dict_item["Perovskite_composition_b_ions_coefficients"] + \
                ", Perovskite composition c ions: " + dict_item["Perovskite_composition_c_ions"] + \
                ", Perovskite composition c ions coefficients: " \
                + dict_item["Perovskite_composition_c_ions_coefficients"] + \
                ", Perovskite additives compounds: " + dict_item["Perovskite_additives_compounds"] + \
                ", Perovskite additives concentrations: " + dict_item["Perovskite_additives_concentrations"] + \
                ", Perovskite band gap: " + dict_item["Perovskite_band_gap"] + \
                ", ETL thickness: " + dict_item["ETL_thickness"] + \
                ", Perovskite deposition solvents: " + dict_item["Perovskite_deposition_solvents"] + \
                ", Perovskite deposition procedure: " + dict_item["Perovskite_deposition_procedure"] + \
                ", Perovskite deposition thermal annealing temperature: " \
                + dict_item["Perovskite_deposition_thermal_annealing_temperature"] + \
                ", Perovskite deposition thermal annealing time: " \
                + dict_item["Perovskite_deposition_thermal_annealing_time"] + \
                ", Perovskite deposition solvents mixing ratios: " \
                + dict_item["Perovskite_deposition_solvents_mixing_ratios"] + \
                ", Perovskite deposition synthesis atmosphere: " \
                + dict_item["Perovskite_deposition_synthesis_atmosphere"] + \
                ", Perovskite deposition aggregation state of reactants: " \
                + dict_item["Perovskite_deposition_aggregation_state_of_reactants"] + \
                ", HTL stack sequence: " + dict_item["HTL_stack_sequence"] + \
                ", HTL additives compounds: " + dict_item["HTL_additives_compounds"] + \
                ", HTL deposition procedure: " + dict_item["HTL_deposition_procedure"] + \
                ", HTL thickness list: " + dict_item["HTL_thickness_list"] + \
                ", Backcontact stack sequence: " + dict_item["Backcontact_stack_sequence"] + \
                ", Backcontact additives compounds: " + dict_item["Backcontact_additives_compounds"] + \
                ", Backcontact deposition procedure: " + dict_item["Backcontact_deposition_procedure"] + \
                ", Backcontact thickness list: " + dict_item["Backcontact_thickness_list"] + \
                ", Cell area measured: " + dict_item["Cell_area_measured"] + \
                ", Cell number of cells per substrate: " + dict_item["Cell_number_of_cells_per_substrate"] + \
                ", Cell architecture: " + dict_item["Cell_architecture"] + \
                ", Cell flexible: " + dict_item["Cell_flexible"] + \
                ", Cell semitransparent: " + dict_item["Cell_semitransparent"] + \
                ", Cell semitransparent wavelength range: " + dict_item["Cell_semitransparent_wavelength_range"] + \
                ", Module: " + dict_item["Module"] + \
                ", JV average over n number of cells: " + dict_item["JV_average_over_n_number_of_cells"] + \
                ", JV light intensity: " + dict_item["JV_light_intensity"] + \
                ", JV light spectra: " + dict_item["JV_light_spectra"]

            if dict_item["JV_default_PCE"] == "":
                continue
            pce_value = float(dict_item["JV_default_PCE"])
            pce_label = "Unknown"
            if 0.0 <= pce_value <= 10.0:
                pce_label = "EX LOW"
            elif 10.0 < pce_value <= 15.0:
                pce_label = "LOW"
            elif 15.0 < pce_value <= 20.0:
                pce_label = "NORMAL"
            elif 20.0 < pce_value <= 25.0:
                pce_label = "HIGH"
            elif pce_value > 25.0:
                pce_label = "TOP"
            completion = \
                "JV default Voc: " + dict_item["JV_default_Voc"] + \
                ", JV default Jsc: " + dict_item["JV_default_Jsc"] + \
                ", JV default FF: " + dict_item["JV_default_FF"] + \
                ", JV default PCE: " + pce_label

            prompt = prompt.replace(": ,", ": Unknown,").replace("nan", "Unknown")
            if prompt.count("Unknown") >= 6:
                continue
            if completion.count("Unknown") >= 6:
                continue
            line = [prompt, completion]
            if pce_label == "EX LOW":
                with open("finetune_classification_EXLOW11.csv", "a", encoding="utf-8", newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(line)
            elif pce_label == "LOW":
                with open("finetune_classification_LOW11.csv", "a", encoding="utf-8", newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(line)
            elif pce_label == "NORMAL":
                with open("finetune_classification_NORMAL11.csv", "a", encoding="utf-8", newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(line)
            elif pce_label == "HIGH":
                with open("finetune_classification_HIGH11.csv", "a", encoding="utf-8", newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(line)
            elif pce_label == "TOP":
                with open("finetune_classification_TOP11.csv", "a", encoding="utf-8", newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(line)


def test_model():
    test_text = pd.read_json('finetune_classification240_test_prepared.jsonl', lines=True)
    voc = []
    voc_hat = []
    jsc = []
    jsc_hat = []
    ff = []
    ff_hat = []
    # pce = []
    # pce_hat = []
    pce_true = []
    pce_pred = []
    label_list = ["EX LOW", "LOW", "NORMAL", "HIGH", "TOP"]

    for i in range(len(test_text)):
        print("No." + str(i))
        try:
            prompt_text = test_text['prompt'][i]
            answer_text = test_text['completion'][i].split("\n")[0].replace(": ,", ": '',")
            res = openai.Completion.create(model=ft_model, prompt=prompt_text, max_tokens=80, temperature=0, stop="\n")
            result_text = res['choices'][0]['text'].replace(": ,", ": '',")
            print("output -> " + result_text)
            print("answer -> " + answer_text)

            answers = answer_text.split(",")
            results = result_text.split(",")
            if len(answers) != len(results):
                print("schema(s) missing")
                continue
            for schema_i in range(len(results)):
                answer_schema = answers[schema_i].split(":")[0].strip()
                result_schema = results[schema_i].split(":")[0].strip()
                if answer_schema != result_schema:
                    print("schema unaligned:" + answer_schema + " <> " + result_schema)
                    continue
                schema_answer = answers[schema_i].split(":")[1].strip()
                schema_result = results[schema_i].split(":")[1].strip()
                if result_schema != "JV default PCE":
                    answer_figure = float(schema_answer)
                    result_figure = float(schema_result)

                    if result_schema == "JV default Voc":
                        voc.append(answer_figure)
                        voc_hat.append(result_figure)
                    elif result_schema == "JV default Jsc":
                        jsc.append(answer_figure)
                        jsc_hat.append(result_figure)
                    elif result_schema == "JV default FF":
                        ff.append(answer_figure)
                        ff_hat.append(result_figure)
                elif result_schema == "JV default PCE":
                    # pce.append(answer_figure)
                    # pce_hat.append(result_figure)
                    pce_true.append(schema_answer)
                    pce_pred.append(schema_result)

            print("______________________________")

        except openai.InvalidRequestError as e1:
            print(str(e1))
            print("______________________________")

    mae_voc = np.mean(np.abs(np.array(voc)-np.array(voc_hat)))
    mae_jsc = np.mean(np.abs(np.array(jsc)-np.array(jsc_hat)))
    mae_ff = np.mean(np.abs(np.array(ff)-np.array(ff_hat)))
    # mae_pce = np.mean(np.abs(np.array(pce)-np.array(pce_hat)))
    print("mae_voc: " + str(mae_voc))
    print("mae_jsc: " + str(mae_jsc))
    print("mae_ff: " + str(mae_ff))
    # print("mae_pce: " + str(mae_pce))
    print("______________________________")

    rmse_voc = np.sqrt(np.mean(np.square(np.array(voc) - np.array(voc_hat))))
    rmse_jsc = np.sqrt(np.mean(np.square(np.array(jsc) - np.array(jsc_hat))))
    rmse_ff = np.sqrt(np.mean(np.square(np.array(ff) - np.array(ff_hat))))
    # rmse_pce = np.sqrt(np.mean(np.square(np.array(pce) - np.array(pce_hat))))
    print("rmse_voc: " + str(rmse_voc))
    print("rmse_jsc: " + str(rmse_jsc))
    print("rmse_ff: " + str(rmse_ff))
    # print("rmse_pce: " + str(rmse_pce))
    print("______________________________")

    mean_voc = np.mean(voc)
    mean_jsc = np.mean(jsc)
    mean_ff = np.mean(ff)
    # mean_pce = np.mean(pce)
    print("mean_voc: " + str(mean_voc))
    print("mean_jsc: " + str(mean_jsc))
    print("mean_ff: " + str(mean_ff))
    # print("mean_pce: " + str(mean_pce))
    print("______________________________")

    var_voc = np.var(voc)
    var_jsc = np.var(jsc)
    var_ff = np.var(ff)
    # var_pce = np.var(pce)
    print("var_voc: " + str(var_voc))
    print("var_jsc: " + str(var_jsc))
    print("var_ff: " + str(var_ff))
    # print("var_pce: " + str(var_pce))
    print("______________________________")

    matrix = confusion_matrix(pce_true, pce_pred, labels=label_list)
    res = classification_report(pce_true, pce_pred, target_names=label_list)
    print(matrix, "\n")
    print(res)


test_model()

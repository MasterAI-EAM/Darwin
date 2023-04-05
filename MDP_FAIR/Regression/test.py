import openai
import pandas as pd


openai.api_key = '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'
ft_model = "davinci:ft-greendynamics-2023-03-29-17-28-16"


def test_model():
    test_text = pd.read_json('finetune_400_prepared_test.jsonl', lines=True)
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

                # todo: compare "schema_answer" and "schema_result"

            print("______________________________")

        except openai.InvalidRequestError as e1:
            print(str(e1))
            print("______________________________")
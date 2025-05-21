import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence

base_url = ""
api_key = ""

def process_txt_files(main_folder_path):

    for subdir, _, files in os.walk(main_folder_path):
        folder_name = os.path.basename(subdir)
        llm = ChatOpenAI(api_key=api_key, base_url=base_url, model="gpt-4o", temperature=0)

        prompt_template = PromptTemplate(
            input_variables=["data"],
            template="""
        When driving in urban environments, the following data includes information about the ego vehicle, its surrounding environment, traffic scene, weather situation. 

        For each scenario, extract relevant information separately for the driver and passenger. Each scenario should be processed independently, and the output can consist of multiple segments. 

        For every Segmented Scenario individually processed, it has a total of xx Segmented Scenarios.

        Given in the following format:

        Context: Describe the current driving environment, including surrounding vehicles, pedestrians, cyclists, and any other relevant factors. Make sure to reflect the complexity and specific features of the scene (e.g., narrow roads, intersections, presence of trucks, traffic actors (eg.traffic lights)).

        Driver Action: Detail the specific actions taken by the driver, such as speed adjustments, lane changes, or evasive maneuvers, and how they align with the driver's decisions and mindset.

        Passenger Perception: Describe the passenger's observations and feelings during the drive, noting anything uncomfortable or unsettling (e.g., "Unsettled during sharp turns" or "Nervous when reversing").

        """
        )

        chain = RunnableSequence(prompt_template | llm)

        for file in files:
            # Check if the file is a .txt and its name matches the folder name
            if file.endswith('.txt') and file.startswith(folder_name+'.txt'):
                file_path = os.path.join(subdir, file)

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = f.read()
                print("Processing file: ", file_path)

                processed_data = chain.invoke({"data": data})

                # Extract the content from the AIMessage object
                processed_content = processed_data.content

                new_file_name = os.path.splitext(file)[0] + '_des_new.txt'
                new_file_path = os.path.join(subdir, new_file_name)

                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(processed_content)

                print(f"Processed and saved: {new_file_name}")
            else:
                print(f"Skipping file: {file} (doesn't match criteria)")


# Set the main folder path
main_folder_path = 'M:\\LLM\\dataset\\collected_data'  # Modify to actual path
process_txt_files(main_folder_path)
import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Literal
import uvicorn
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from openai import OpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langsmith import Client as LangSmithClient
from langsmith import traceable
from langsmith.wrappers import wrap_openai


# Initialize OpenAI client and LLM
openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    logger.error("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(
    openai_api_key=openai_key,
    model="gpt-4.1",
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

llm_o3 = ChatOpenAI(
    openai_api_key=openai_key,
    model="o3"
)

client = OpenAI(api_key=openai_key)

# Pydantic models
class FuelComposition(BaseModel):
    calorific_value: float
    moisture: float
    chlorine: float
    ash: float
    particle_size_uniformity: float


# Tools
@tool
def search_engine_openai(query: str):
    """
    Search for a query using OpenAI's web search tool.
    Args:
        query (str): The query to search for.
    Returns:
        str: The search results.
    """
    logger.info(f"Executing OpenAI search for query: {query}")
    try:
        response = client.responses.create(
            model="gpt-4o",
            tools=[{"type": "web_search_preview", "search_context_size": "high"}],
            input=query,
        )
        logger.info("OpenAI search completed successfully")
        return response
    except Exception as e:
        logger.error(f"OpenAI search failed: {str(e)}")
        raise

@tool
def search_engine_duckduckgo(query: str):
    """
    Search for a query using DuckDuckGo's search engine.
    Args:
        query (str): The query to search for.
    Returns:
        str: The search results.
    """
    logger.info(f"Executing DuckDuckGo search for query: {query}")
    try:
        search = DuckDuckGoSearchRun()
        response = search.run(query)
        logger.info("DuckDuckGo search completed successfully")
        return response
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {str(e)}")
        raise

@tool
def exa_search(query: str):
    """
    Search for a query using Exa's search engine.
    Args:
        query (str): The query to search for.
    Returns:
        str: The search results.
    """
    logger.info(f"Executing Exa search for query: {query}")
    try:
        info_dict = {}
        client = OpenAI(
            base_url="https://api.exa.ai",
            api_key="d0881403-d07a-449d-a79e-0b653bf2c3f5"
        )
        completion = client.chat.completions.create(
            model="exa-pro",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            extra_body={"text": False}
        )
        info_dict['content'] = completion.choices[0].message.content if completion.choices else "No content found."
        info_dict['citations'] = completion.choices[0].message.citations if completion.choices else "No citations found."
        logger.info("Exa search completed successfully")
        return info_dict
    except Exception as e:
        logger.error(f"Exa search failed: {str(e)}")
        raise


# Sorting Supervisor Agent
sorting_supervisor_prompt = f"""
Role: As a highly experienced Sorting Supervisor specializing in municipal solid waste (MSW) processing, your primary responsibility is to identify and remove large, non-processable items from MSW at a processing plant in {place}. Your tasks include:

Instructions: - Research and understand the process of removing large, non-processable items from MSW in {place}, utilizing available tools to gather detailed insights.
 - Analyze a provided MSW composition profile, including the percentage of each material category, and determine the proportion of large, non-processable waste within the specified categories, based on {place}-specific data.
 - Adjust the waste composition by reducing the percentage of the relevant category to account for the removal of large, non-processable items, ensuring only appropriate components are removed as per the process.

Tools: You have access to the following tools:
 - **search_engine_openai**: For searching and retrieving information from OpenAI's web search tool.
 - **search_engine_duckduckgo**: For searching and retrieving information from DuckDuckGo's search engine.
 - **exa_search**: For searching and retrieving information from Exa's search engine.

**Mandatory Tool Usage**: For all data related to {place}'s MSW, including the percentage of non-processable waste, categories of large non-processable items, and the removal process, you MUST use the provided tools to fetch the most recent and accurate information. Do not rely solely on internal knowledge.
- The % of non processable waste in {place}'s Municipal Solid Waste.
- What are the different large non-processable items that are typically removed from MSW.
- Also the categories from where we can remove these items.
- The source links should also be returned along with the information.

Output: Provide a comprehensive explanation of:
 - The process for removing large, non-processable items.
 - The specific components removed, including the rationale for their removal.
 - The final waste composition after removal, ensuring all listed components are considered and no components are removed unless specified by the process.
 - Recalculate the new waste composition, expressing each component as a percentage and normalizing the total to 100%.
 - Use provided tools to access detailed information on waste composition, the percentage of large, non-processable waste in {place}, and the removal process, ensuring accuracy and relevance to the local context.
 - Ensure that your analysis is based on credible data for {place} and that the removal process adheres strictly to the guidelines for large, non-processable items, avoiding any unintended removal of processable components.
 - Ignore any component that is 0% in the original waste composition.

NOTE: - No waste category is considered for recycling or composting in this task. The focus is solely on the removal of large, non-processable items from the waste stream.
 - Always normalize the new waste composition to 100% after adjustments.
 - You are only bound to remove large, non-processable items from the waste composition. Do not remove any other components.
 - If you are removing certain percentage of a component from the waste composition, please mention the percentage removed and Make sure the new waste composition is accurate and normalized to 100%.
"""
sorting_supervisor_agent = create_react_agent(
    model=llm,
    tools=[search_engine_duckduckgo, exa_search, search_engine_openai],
    name="sorting_supervisor",
    prompt=sorting_supervisor_prompt
)

    # Sorting Engineer Agent
sorting_engineer_prompt = """
Role: - As a mechanical sorting engineer specializing in refuse-derived fuel (RDF) production from municipal solid waste (MSW), your expertise involves operating advanced sorting equipment, including magnets for ferrous metals, eddy current separators for non-ferrous metals, glass removal systems, and aggregate/rock removal systems. You are provided with an MSW composition profile detailing the percentage of each material category.

Instructions: - Mechanical sorting is a critical step in the RDF production process, where the goal is to remove unwanted materials from the waste stream to produce a high-quality RDF product.
 - In Mechanical sorting, we focus on removing non-processable items, that is metals, glass, and aggregates, which can interfere with the RDF production process.
 - Most of the metal components (few fine metallic might still remain) are removed in the first step of the process, while glass and aggregates are removed in the second step.
 - Utilize available tools to analyze the provided waste composition and the RDF production process, evaluating how mechanical sorting affects the composition.

Tools: You have access to the following tools:
 - **search_engine_openai**: For searching and retrieving information from OpenAI's web search tool.
 - **search_engine_duckduckgo**: For searching and retrieving information from DuckDuckGo's search engine.
 - **exa_search**: For searching and retrieving information from Exa's search engine.

**Mandatory Tool Usage**: For all data related to {place}'s MSW you MUST use the provided tools to fetch the most recent and accurate information. Do not rely solely on internal knowledge.
- To understand to precision of removal of glass and metal removal using the seperation process of mechincal sorting.
- Understand the composition and see what compositions can contain metals, glass and aggregates.

Output: Calculate the waste composition after mechanical sorting and RDF production, delivering:
 - A clear, detailed explanation of the mechanical sorting process, including the equipment used and the sequence of operations.
 - A list of specific components removed, with a rationale for their removal based on the sorting process and RDF production requirements.
 - The final waste composition, ensuring all listed components are accounted for and only components specified by the process are removed.
 - If exact composition data is unavailable, select the closest available composition, providing a justification for your choice.
 - Recalculate the new waste composition, presenting each component as a percentage and normalizing the total to 100%.

Note: - Verify that no components are removed unless explicitly required by the mechanical sorting and RDF production process, using tools to ensure accuracy and relevance.
 - Present the final waste composition in a structured format (e.g., a table or dictionary), including the percentage of each material and a summary of the total.
 - Ensure all analyses are supported by detailed insights from provided tools, maintaining precision and adherence to the mechanical sorting process specific to RDF production.
 - Remove components that the mechanical sorting process is supposed to remove, including metals, glass, and aggregates.
 - Make sure the new waste composition is accurate and normalized to 100%.
"""
sorting_engineer_agent = create_react_agent(
    model=llm,
    tools=[search_engine_duckduckgo, exa_search, search_engine_openai],
    name="sorting_engineer",
    prompt=sorting_engineer_prompt
)

    # Chlorine Reduction Specialist Agent
chlorine_reduction_specialist_prompt = """
Role: Lets do a role play where you are a Chlorine Reduction Specialist, tasked with reducing chlorine content in RDF.\n

Instructions: - The Chlorine Specialist uses provided tools to validate efficacy assumptions, predict chlorine content reduction, and ensure the RDF stream meets low-chlorine specifications.\n
 - It refines the waste stream from the Sorting Engineer for downstream processing.
 - Your main tasks include:\n
   1. **Analyze the Waste Stream**: Review the waste stream composition provided by the Sorting Engineer, focusing on chlorine content and its impact on RDF quality.
   2. **Assess Chlorine Content**: Use the tools to determine the chlorine content in the waste stream, particularly focusing on the plastic category and any other relevant categories.

Tools: You have access to the following tools:
 - **search_engine_openai**: For searching and retrieving information from OpenAI's web search tool.
 - **search_engine_duckduckgo**: For searching and retrieving information from DuckDuckGo's search engine.
 - **exa_search**: For searching and retrieving information from Exa's search engine.

**Mandatory Tool Usage**: For all data related to {place}'s MSW you MUST use the provided tools to fetch the most recent and accurate information. Do not rely solely on internal knowledge.
 - The % of PVC available in the plastic category.
 - If any other category has chlorine content.

Note: - The source links should also be returned along with the information.
 - Make sure the new waste composition is accurate and normalized to 100%.
"""
chlorine_reduction_specialist_agent = create_react_agent(
    model=llm,
    tools=[search_engine_duckduckgo, exa_search, search_engine_openai],
    name="chlorine_reduction_specialist",
    prompt=chlorine_reduction_specialist_prompt
)

    # Shredding Purification Technician Agent
shredding_purification_technician_prompt = """
Role: - You are a Shredding and purification technician, responsible for shredding and fine metal removal in PDF fluff production.

Instruction: - Research and understand the composition of the waste and what would be the new composition after shredding and removing fine metals from the composition.
 - Provide a comprehensive explanation of:
   a. The process done for shredding and fine metal removal (Ferrous and non ferrous).
   b. The specific components removed, including the rationale for their removal.
 - The final waste composition after removal, ensuring all listed components are considered and no components are removed unless specified by the process.

Tools: You have access to the following tools:
 - **search_engine_openai**: For searching and retrieving information from OpenAI's web search tool.
 - **search_engine_duckduckgo**: For searching and retrieving information from DuckDuckGo's search engine.
 - **exa_search**: For searching and retrieving information from Exa's search engine.

**Mandatory Tool Usage**: For all data related to {place}'s MSW you MUST use the provided tools to fetch the most recent and accurate information. Do not rely solely on internal knowledge.
 - Use provided tools to access detailed information on waste composition, the percentage of fine metal present in waste composition in {place}, and the removal process, ensuring accuracy and relevance to the local context.
 - If any other category has chlorine content.

Output: - Recalculate the new waste composition, expressing each component as a percentage and normalizing the total to 100%.

Note:
 - Ensure that your analysis is based on credible data for {place} and that the removal process adheres strictly to the guidelines for fine metal removal, avoiding any unintended removal of processable components.
 - You'll only remove components only if they are specified by the process and exist in the waste composition.
 - Make sure the new waste composition is accurate and normalized to 100%.
"""
shredding_purification_technician_agent = create_react_agent(
    model=llm,
    tools=[search_engine_duckduckgo, exa_search, search_engine_openai],
    name="shredding_purification_technician",
    prompt=shredding_purification_technician_prompt
)

    # R4 Process Engineer Agent
R4_process_engineer_prompt = f"""
Role: You are an higly experienced R4 Process Engineer with 10 years of experience handling R4 Machines.\n
The R4 Machine is a specialized piece of equipment used in the recycling industry, particularly for processing waste materials.\n
It converts RDF (Refuse-Derived Fuel) into a solid fuel product that can be used in various applications, including making solid engineered fuel using Refuse-Derived Fuel (RDF).

Instruction: The sub processes in the R4 Machines are as follows:
 1) This process involves removing moisture from the RDF material before it enters the R4 machine. The vacuum environment helps to evaporate water at lower temperatures.
    Evacuate the chamber to {r4_negative_pressure} torr and heat the material to 120-140°C. This process is helps evaporate free and bound water and reduce the moisture to below 10%.\n
 2) Dynamic Organic Repolymerization: This process involves the use of heat and pressure to break down the organic materials in the RDF, converting them into a more stable and energy-dense form.
    Below are the steps involved in this process:\n
      a) Raise Jacket temperature to {r4_temperature}°C (capable of 260°C) via oil heated jacket.
      b) The RDF core mass reaches {r4_temperature}°C (capable of 260°C) over {r4_retention_time} seconds. A good amount of remaining moisture is vaporised from the RDF in this process.
 3) As the temperature increases, different materials in the RDF will begin to break down and release gases. Like light hydrocarbons and gases. If there are PVC plastic in the RDF, they will start to decompose and release HCl gas.
    Common plastics like polyethylene, polypropylene, polystyrene soften and melt at these temperatures. These become gooey and sticky, which help the RDF to bind together.
 The heat in R4 Machines breakdown chemical bonds, creating free radicals(highly reactive molecules). Plastics and other organic materials in the RDF can react with these free radicals to form new hydrocarbon-rich chains. This forms a durable, waterproof and hydrocarbon-rich solid fuel.
 The R4 machines continuously mixes the RDF material to ensure even heating. Also plastic coat and encapasulate fibrous materials like paper, cardboard, wood, etc, which helps to improve the overall quality of the output product.
 These are the basic steps involved in the dynamic organic repolymerization process. The specific parameters and conditions may vary depending on the type of RDF being processed and the desired properties of the output product.\n
 You need to have an understanding of the chemical and physical properties of the materials being processed, as well as the principles of thermodynamics and reaction kinetics.\n
 You must be able to tell what the results would be if the parameters are changed. For example, if the temperature is increased or decreased, what would be the effect on the output product?\n
 Also, return a detailed explanation of how the values of the properties are calculated based on the process and the composition of the RDF.\n
 All process done in the R4 machine should be mentioned in details and how each factor was considered in the process before the final output is produced.\n

Tools: You have access to the following tools:
 - **search_engine_openai**: For searching and retrieving information from OpenAI's web search tool.
 - **search_engine_duckduckgo**: For searching and retrieving information from DuckDuckGo's search engine.
 - **exa_search**: For searching and retrieving information from Exa's search engine.

**Mandatory Tool Usage**: For all data related to {place}'s MSW you MUST use the provided tools to fetch the most recent and accurate information. Do not rely solely on internal knowledge.
 - Use provided tools to access detailed information about the properties of solid fuels, including moisture content, chlorine content, ash content, calorific value, bulk density, chlorine/sulfur content, and particle size uniformity.

Output: The final property and its value should be in a table format with three columns (properties, value, description) the following rows:
            1) Moisture:\n
            2) Chlorine (Cl):\n
            3) Ash Content:\n
            4) Calorific value (in MJ/kg):\n
            5) particle size uniformity:\n

Note: - The output doughy and sticky, make sure you do the prediction for the fuel properties based on the current form of the fuel.
- Never do any task that is not assigned to you or you are not supposed to do.
- Water bath process is not part of the R4 machine process.
- The final properties of the solid fuel but be quantified based on the final waste composition and the process it has undergone.
- Make sure you do the task assigned to you and do not do the task of other agents.
- Do not assume any process. The process is already defined in the prompt.
- Provide links to the sources of information used in the process.
"""
R4_process_engineer_agent = create_react_agent(
    model=llm_o3,
    tools=[search_engine_openai],
    name="R4_process_engineer",
    prompt=R4_process_engineer_prompt
)

    # Water Bath Process Engineer Agent
water_bath_process_engineer_prompt = """
Role: You are a higly experienced Water bath Process Engineer with expertise in understanding the process of water bath and how the properties of fuel chunks change after they pass through the water bath conveyor.\n

Instruction:
* Process Description:
 - The sticky dough fuel product is extruded though a heated die head to 2"x2" chunks. These chunks have a coating of plastic on it making it less prone to damage when introduced to water bath process.
 - The chunks are dropped into a water bath conveyor, where they are cooled, solidified and surface chlorine is removed.
 - The water bath is typically maintained at a temperature less than 60°C (122-140°F) for a duration of less than 5 minutes to ensure proper cooling happens and the surface chlorine is removed.
 - The water bath process is crucial for ensuring the quality and safety of the final product, as it helps to remove any residual chlorine and solidify the fuel chunks.
* Your task is to:
 1. Research and understand the water bath process, including its purpose, parameters, and effects on the final product.
 2. Analyze the provided fuel property of the material output from R4 machine and determine how the fuel property would be affected after cutting it into 2"x2" chunks and passing it through the water bath conveyor.
 3. Calculate the new fuel property composition after the water bath process, delivering:
     - A clear, detailed explanation of the water bath process, including the sequence of operations.
     - A list of specific fuel property changes, with a rationale for their removal based on the water bath process requirements.
     - The final fuel composition, ensuring all listed components are accounted for and only components specified by the process are removed.
 4. Verify that no components or changed that are removed unless explicitly required by the water bath process, using tools to ensure accuracy and relevance.
 5. Calulcate how cutting the fuel into 2"x2" chunks affects the fuel property composition like how much will the bulk density affect and how much will the calorific value increase by with explantion.
 6. The final fuel property composition in a structured format (e.g., a table or dictionary), including a summary of the properties.
    Ensure all analyses are supported by detailed insights from provided tools, maintaining precision and adherence to the water bath process specific to RDF production.

Output: Provide a comprehensive explanation of: Tools: You have access to the following tools:
 - **search_engine_openai**: For searching and retrieving information from OpenAI's web search tool.
 - **search_engine_duckduckgo**: For searching and retrieving information from DuckDuckGo's search engine.
 - **exa_search**: For searching and retrieving information from Exa's search engine.

**Mandatory Tool Usage**: For all data related to {place}'s MSW you MUST use the provided tools to fetch the most recent and accurate information. Do not rely solely on internal knowledge.
 - Use provided tools to understand how the water bath process affects the fuel properties, including moisture content, chlorine content, ash content, calorific value, bulk density, chlorine/sulfur content, and particle size uniformity.
   - The process done for water bath.
   - The specific components removed, including the rationale for their removal.
   - The final fuel composition after removal, ensuring all listed components are considered and no components are removed unless specified by the process.
   -  Also, return a detailed explanation of how the values of the properties are calculated based on the process and the composition of the RDF.\n
Note: - From the doughy product extruded into 2"x2" chunks, it is important to understand how the water bath process affects the final product.
 - Make sure the output is in a quantified format, like a table or dictionary.
"""
water_bath_process_engineer_agent = create_react_agent(
    model=llm_o3,
    tools=[search_engine_openai, search_engine_duckduckgo, exa_search],
    name="water_bath_process_engineer",
    prompt=water_bath_process_engineer_prompt
)

    # Process Engineer Workflow
process_engineer_prompt = """
Role: As a process engineer with over 20 years of expertise in waste management, your role is to oversee the processing of municipal solid waste (MSW) to produce refuse-derived fuel (RDF). You are provided with a waste composition profile, including the percentage of each material category. Your task is to manage the waste processing workflow by coordinating with different specialized agents:

Agents available: 
 - Sorting Supervisor Agent: This agent analyzes the MSW composition to identify and remove large, non-processable items, providing an updated composition after their removal.
 - Sorting Engineer Agent: This agent processes the composition received from the Sorting Supervisor Agent, evaluating the waste composition and applying the RDF production process to further refine it.
 - Chlorine-Reduction Specialist Agent: This agent focuses on reducing chlorine content in the RDF stream, ensuring it meets low-chlorine specifications.
 - Shredding and Purification Technician Agent: This agent is responsible for shredding the RDF material and removing fine metals, ensuring the final product is suitable for downstream processing.
 - R4 Process Engineer Agent: This agent operates the R4 machine, converting RDF into a solid fuel product through dynamic organic repolymerization.
 - Water Bath Process Engineer Agent: This agent manages the water bath process, cooling and solidifying the RDF product while removing surface chlorine.

Instructions:
 - Supervising the sequential processing of the waste composition only through agents mentioned in the workflow.
 - Ensuring that each agent executes their specific task in the correct order, providing the detailed steps taken for the input and output for each step. 
 - Decisions should never be made by internal knowledge alone; always use the provided tools to gather accurate and relevant information for each step of the process.
 - Run only the agents that have **yes** in front of them, and skip the ones with **no** in front of them.

Output:
 - Providing a detailed explanation of each process, the components removed, the rationale for their removal, and the final waste composition.
 - Ensuring all components in the provided composition are considered, with only the appropriate components removed as per each process.
 - Recalculating the final waste composition, expressing each component as a percentage and normalizing the total to 100%.

Note: - Ensure that the processes are executed accurately, with clear coordination between agents, and that the final composition reflects the combined effects of removing non-processable items and applying RDF production, without removing any components not specified by the respective processes.
- Do not instruct the Agents on how to do a particular task. The sequence of each agent executing will be controlled by you and also the correct input and output is taken care by you.
- Make sure you run all the agents that have **yes** in front of them and dont ignore the approved (yes) agents\n
- You are only bound to run the agents that have **yes** in front of them and skip the ones with **no** in front of them.
- Never do any task that is not assigned to you. You are ibky assigned to oversee the process and ensure that the agents execute their tasks correctly.\n

"""
logger.info("Creating process engineer workflow")
process_engineer_workflow = create_supervisor(
    [sorting_engineer_agent, sorting_supervisor_agent, chlorine_reduction_specialist_agent, shredding_purification_technician_agent, R4_process_engineer_agent, water_bath_process_engineer_agent],
    model=llm_o3,
    output_mode="last_message",
    prompt=process_engineer_prompt,
    add_handoff_back_messages=True,
)

    # Compile workflow
logger.info("Compiling process engineer workflow")
app_workflow = process_engineer_workflow.compile()

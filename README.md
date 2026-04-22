# Sewer Defects Descriptive Analysis

---

## Project overview

The diagram below provides an overview of the project workflow, which is organized into four main steps and one optional component. This repository corresponds to the defect distribution analysis step, highlighted in red in the figure.

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 40, 'rankSpacing': 50}}}%%
flowchart LR

    A[Database Structure]:::data -->  B[Validation<br/>Rules]:::validation
    B --> D[Defect Distribution<br/>Analysis]:::analysis
    D --> E[Defect-Factor<br/>Correlation]:::analysisP2

    %% Optional branch BELOW
    subgraph optional_branch[ ]
        direction TB
        C[Optional: Validation<br/>Report]:::optional
    end
    style optional_branch fill:none,stroke:none

    subgraph stage1[ ]
        A[Defect Distribution<br/>Analysis]
    end
 
    style stage1 stroke:#FF0000,stroke-width:3px,fill:none
    B -.-> C

    %% Click links
    click A "https://github.com/SewerDefectAnalysis/Database_Structure"
    click B ""
    click C ""

    %% Styles
    classDef data fill:#E8F0FE,stroke:#1A73E8,stroke-width:1.8px
    classDef validation fill:#E6F4EA,stroke:#188038,stroke-width:1.8px
    classDef analysis fill:#F3E8FD,stroke:#9334E6,stroke-width:1.8px
    classDef analysisP2 fill:#FDEBD0,stroke:#E67E22,stroke-width:1.8px
    classDef prediction fill:#FEF7E0,stroke:#F9AB00,stroke-width:1.8px
    classDef optional fill:#F5F5F5,stroke:#666666,stroke-width:1.8px,stroke-dasharray: 5 5

```

## 1. Defect Distribution Analysis - Repository

This repository contains the code developed for the paper *Distribution and Properties of Defects in Urban Drainage Systems: An Analysis of Auckland's Sewer Network* by Juana Herrán, María A. González, Jakobus E. van Zyl, and Theunis F. P. Henning.

The project uses sewer pipe and inspection data to perform a descriptive analysis of defects and their properties. The workflow enables users to:

* Characterize a sewer network and compare it with the subset of pipes inspected via CCTV, assessing representativeness.
* Analyze the distribution of defects per pipe and the distribution of defect types across different materials.
* Explore and visualize defect properties, including:
  * Defect size
  * Longitudinal distance
  * Extent
  * Circumferential position

The implemented methodology is generalizable and can be applied to sewer datasets from different systems.

This repository is part of a broader project, as described earlier. It is intended to be used after implementing the database structure and applying the validation rules. To support this workflow, an example dataset (`EXAMPLE_DATA_Database.xlsx`) is provided in the [Database Structure Repository](https://github.com/SewerDefectAnalysis/Database_Structure). This dataset was used to run the previous steps and generate the validated dataset (`VALIDATED_DATA.xlsx`) included in this repository.

---
## 2. Code Structure

The codebase consists of eight Python files and one Jupyter Notebook, where the full analysis workflow is executed. The core logic is implemented in the Python scripts, while the Jupyter Notebook is used to orchestrate their execution and visualize results. Below is a general description of each file. Detailed documentation of individual functions can be found within the corresponding source files.

#### `Run_defect_description.ipynb`
This Jupyter Notebook serves as the main entry point of the project. It orchestrates the execution of all scripts and displays the tables and figures generated during the analysis. The notebook does not contain core processing logic; instead, it focuses on configuration, execution flow, and result visualization. It also manages the selection of materials (e.g., AC, CONC, VC, PVC, PE) and factors (e.g., age, length, slope, etc.) to be included in the analysis.
#### `config.py`
Defines the global variables that are accessed and used by the other files.
#### `load_excel.py`
Contains functions for loading and importing input data from an Excel file.
#### `data_preparation.py`
Prepares the data for analysis. It performs the required data merges and includes functions to select and validate the chosen materials and factors. In addition, it defines color maps for materials and defect types.
#### `defect_length_calculation.py`
Normalizes the longitudinal distance of each defect using pipe length and calculates defect length based on continuous defects.
When defects of the same type within a pipe are marked with start (`S#`) and finish (`F#`), the length is computed as the difference between their normalized positions. Otherwise, defects are treated as single points with zero length.This is used to represent defect extent along the pipe and supports subsequent analyses and visualizations.
#### `dataset_description.py`
Provides the functions for a detailed description of the dataset, summarizing key properties and characteristics of the sewer network and inspections. It also analyzes the representativeness of the pipes inspected by CCTV.
#### `defect_general_description.py`
Includes functions that generate a general description of the network, such as the number of defects per kilometer and per pipe, as well as the distribution of defect types by material.
#### `defect_properties_description.py`
Provides functions to analyze the properties of observed defects, including size, longitudinal distance, extent, and clock reference position.
#### `defect_correlation.py`
Contains the function used to calculate pearson correlations between defect types.

---
## 3. Input Data

### 3.1 How to Prepare the Data

You can proceed in two ways:

- **Option 1 — Create your own input file**  
  Manually create an Excel file that follows the expected structure and naming conventions.


- **Option 2 — Follow the full framework (recommended)**  
  Generate the input dataset by following the complete workflow proposed in this project.  
  Each stage of the framework is implemented in a dedicated repository and can be followed step by step:
  - Implement the database structure  
  - Apply the validation rules to your raw data  
  - Export the validated dataset as an Excel file
  > **Important note:** See the workflow diagram above. Each step is clickable and links to the corresponding code and documentation.
  
---

### 3.2 Required Sheets

The input Excel file should contain the following sheets:

1. **PIPES:** Description of the pipes in the network.

2. **CCTV:** Data related to pipe inspections.

3. **DEFECTS:**   Details of observed defects.

4. **HYDRAULIC_PROPERTIES:** *(optional)* Information on the hydraulic characteristics of the pipes, such as flow rate and velocity. 
This sheet should only be included if hydraulic properties are required as part of the network description.

---

### 3.3 Required Columns

Below is a description of the required columns for each sheet when following Option 1 (using a pre-prepared input dataset). These columns must be included to ensure compatibility with the analysis workflow.

If there is any uncertainty regarding the structure or format, users are encouraged to refer to the provided example file (`VALIDATED_DATA.xlsx`).

<table style="text-align:center;">
  <tr>
    <th>Sheet</th>
    <th>Required Column</th>
    <th>Description of the Column</th>
  </tr>

  <!-- PIPES -->
  <tr>
    <td rowspan="3" style="vertical-align:middle;">PIPES</td>
    <td>Pipe_ID</td>
    <td>Unique identifier for each pipe in the network.</td>
  </tr>
  <tr>
    <td>Material</td>
    <td>Pipe material.</td>
  </tr>
  <tr>
    <td>Factors (multiple columns)</td>
    <td>Pipe characteristics such as installation year, diameter, length, and depth. Each attribute should be provided in a separate column.</td>
  </tr>

  <!-- Separator -->
  <tr>
    <td colspan="3" style="border-top: 2px solid #d0d7de;"></td>
  </tr>

  <!-- CCTV -->
  <tr>
    <td rowspan="4" style="vertical-align:middle;">CCTV</td>
    <td>Pipe_ID</td>
    <td>Unique identifier linking each CCTV inspection to the corresponding pipe.</td>
  </tr>
  <tr>
    <td>Inspection_ID</td>
    <td>Unique identifier for each CCTV inspection record, used to link defects and a specific inspection event.</td>
  </tr>
  <tr>
    <td>Inspection_direction</td>
    <td>Indicates whether the survey was carried out upstream or downstream.</td>
  </tr>
  <tr>
    <td>Survey_length</td>
    <td>Length of the pipe surveyed during the CCTV inspection (m).</td>
  </tr>

  <!-- Separator -->
  <tr>
    <td colspan="3" style="border-top: 2px solid #d0d7de;"></td>
  </tr>

  <!-- DEFECTS -->
  <tr>
    <td rowspan="9" style="vertical-align:middle;">DEFECTS</td>
    <td>Defect_ID</td>
    <td>Unique identifier for each defect.</td>
  </tr>
  <tr>
    <td>Pipe_ID</td>
    <td>Unique identifier for each pipe in the network.</td>
  </tr>
  <tr>
    <td>Inspection_ID</td>
    <td>Unique identifier for each CCTV inspection record, used to link defects and a specific inspection event.</td>
  </tr>
  <tr>
    <td>Defect_code</td>
    <td>Type of defect.</td>
  </tr>
  <tr>
    <td>Quantification</td>
    <td>Size of the defect (S, M, or L).</td>
  </tr>
  <tr>
    <td>Continuous_defect</td>
    <td>Indicates whether a defect is part of a continuous segment, using start (S#) and finish (F#) markers.</td>
  </tr>
  <tr>
    <td>Circumferential_start</td>
    <td>Clock position (1–12) where the defect begins.</td>
  </tr>
  <tr>
    <td>Circumferential_end</td>
    <td>Clock position (1–12) where the defect ends.</td>
  </tr>
  <tr>
    <td>Longitudinal_distance</td>
    <td>Position of the defect along the pipe from the upstream manhole (starting point for extended defects).</td>
  </tr>

  <!-- Separator -->
  <tr>
    <td colspan="3" style="border-top: 2px solid #d0d7de;"></td>
  </tr>

  <!-- HYDRAULIC PROPERTIES -->
  <tr>
    <td rowspan="2" style="vertical-align:middle;">HYDRAULIC PROPERTIES</td>
    <td>Pipe_ID</td>
    <td>Unique identifier linking each property value to the corresponding pipe.</td>
  </tr>
  <tr>
    <td>Hydraulic properties (multiple columns)</td>
    <td>Examples include velocity, flow rate, and capacity. Each property should be in a separate column.</td>
  </tr>

</table>


> **Data Assumptions and Notes:** 
>- Pipe identifiers must be consistent across the PIPES, CCTV, and DEFECTS sheets.
>- Defect sizes (Quantification) are expected to be categorized as S, M, or L.
>- Clock reference positions must be integers between 1 and 12.

  

---
## 4. Installation and Setup


Before running the code, ensure you are using Python 3.10 or higher, and install the following requirements: 


- **4.1 Option 1 — Using `requirements.txt` (recommended)**

```bash
pip install -r requirements.txt
```
- **4.2 Option 2 — Manual Installation**

  - `numpy`: Numerical computations
  - `pandas`: Data manipulation
  - `matplotlib`: Plotting and visualization
  - `seaborn`: Statistical plotting
  - `openpyxl`: Excel file support

You can install all required packages using pip:

```python
pip install numpy pandas matplotlib seaborn openpyxl
```

---
## 5. How to run
To execute the analysis, open the notebook `Run_defect_description.ipynb.`

In the _Load Data_ section of the notebook, update the following line with the correct path to the input Excel file and adjust the sheet names if needed. The input file must contain separate sheets for pipes, inspections, and defects.

```python
df_information = load_multiple_sheets(
    r"..\2.Data_validation\Validation_rules\Validated_data.xlsx",
    sheet_names=["PIPES", "CCTV", "DEFECTS", "HYDRAULIC_PROPERTIES"]
)
```
If no hydraulic data are available, this sheet can be omitted and its name removed from the list.



---
## 6. Outputs
The code generates:

- A dataset description, including summary tables and a comparison between all pipes in the network and those inspected via CCTV.

- An analysis of the average number of defects per pipe for the analyzed pipe materials.

- The distribution of defect types by material.

- Plots illustrating the distribution of defect properties by material.

---
## 7. Citation

If you use this repository in your research, please cite the corresponding paper:

Herrán, J., González, M. A., van Zyl, J. E., & Henning, T. F. P. (2026). 
Distribution and Properties of Defects in Urban Drainage Systems: An analysis of Auckland's Sewer Network. _Journal of Water Resources Planning and Management_.

---
## 8. License

This project is distributed under the MIT License.
See the `LICENSE` file for the full text.

---
## 9. Contact

For questions, feedback, or collaboration inquiries related to the paper or this repository, please contact the corresponding author:

**Juana Herrán**  
Email: _jher924@aucklanduni.ac.nz_  
Affiliation: University of Auckland

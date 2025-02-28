# DepenSage

*Smart expense tracking with neural network classification*

## Overview

DepenSage (from French "dépense" meaning expense + "sage" meaning wise) is an intelligent expense tracking tool that automates the process of categorizing credit card transactions and updating your financial spreadsheets.

Unlike traditional expense trackers that rely on fixed rules, DepenSage uses neural networks to learn your personal categorization patterns from historical data, achieving higher accuracy and adapting to your unique financial organization.

## Features

- **Neural Network Classification**: Automatically categorizes transactions based on your historical data
- **Multi-level Classification**: Predicts both main categories and subcategories
- **Google Sheets Integration**: Seamlessly updates your existing Google Sheets expense tracker
- **Hebrew Language Support**: Full support for Hebrew spreadsheets and locality settings
- **CSV Statement Parsing**: Processes standard credit card statement CSV files
- **Automated Sheet Management**: Creates monthly sheets as needed
- **Command-line Interface**: Easy to use for regular statement processing

## Installation

```bash
pip install depensage
```

## Requirements

- Python 3.7+
- Google Sheets API credentials
- TensorFlow 2.4+

## Setup

1. **Google Sheets API Setup**:
   - Create a Google Cloud Project
   - Enable the Google Sheets API
   - Create a service account and download credentials
   - Share your expense spreadsheet with the service account email

2. **Configuration**:
   ```bash
   depensage configure --spreadsheet-id YOUR_SPREADSHEET_ID --credentials-file path/to/credentials.json
   ```

## Usage

### Training the Model

Before first use, train the neural network on your historical data:

```bash
depensage train
```

### Processing Credit Card Statements

Process your credit card statements:

```bash
depensage process your_statement.csv
```

For multiple statements (e.g., yours and your spouse's):

```bash
depensage process your_statement.csv spouse_statement.csv
```

## Project Structure

DepenSage is organized into several logical components:

- **classifier**: Neural network-based classification system
- **sheets**: Google Sheets interaction
- **engine**: Main processing engine
- **config**: Configuration management

## Development

### Running Tests

```bash
python -m unittest discover
```

### Building from Source

```bash
git clone https://github.com/yourusername/depensage.git
cd depensage
pip install -e .
```

## License

MIT License

## Credits

Developed by Itamar Rosenfeld Rauch

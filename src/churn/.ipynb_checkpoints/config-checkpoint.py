{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7f6df7c0-dbec-4c01-8720-e540eab96b3a",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "import yaml"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4eb5075d-21c9-41dd-9958-3073686de475",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_config(path):\n",
    "    \"\"\"Read a YAML config file and fill in default values for anything it leaves out\"\"\"\n",
    "    path = Path(path).resolve()\n",
    "    with open(path, \"r\", encoding = \"utf-8\") as f:\n",
    "        user_settings = yaml.safe_load(f)\n",
    "\n",
    "    cfg = {\n",
    "        \"id_column\": None,\n",
    "        \"drop_columns\" : [],\n",
    "        \"numeric_from_text\" : [],\n",
    "        \"zero_means_missing\" : [],\n",
    "        \"test_size\" : 0.2,\n",
    "        \"random_state\" : 42,\n",
    "    }\n",
    "\n",
    "    cfg.update(user_settings)\n",
    "\n",
    "    cfg[\"_root\"] = str(path.parent.parent)\n",
    "    return cfg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "780194c8-ddec-4139-b5d4-c80591399785",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

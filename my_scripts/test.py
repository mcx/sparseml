from sparseml.transformers.finetune.text_generation import run_train

model = "Xenova/llama2.c-stories15M"
teacher_model = "Xenova/llama2.c-stories15M"
dataset_name = "open_platypus"
concatenate_data = False
output_dir = "./output_finetune"
recipe = "test_trainer_recipe.yaml"
num_train_epochs=2
overwrite_output_dir = True
splits = {
    "train": "train[:50%]",
}

run_train(
    model_name_or_path=model,
    distill_teacher=teacher_model,
    dataset_name=dataset_name,
    output_dir=output_dir,
    recipe=recipe,
    num_train_epochs=num_train_epochs,
    overwrite_output_dir=overwrite_output_dir,
    concatenate_data = concatenate_data,
    splits = splits
)

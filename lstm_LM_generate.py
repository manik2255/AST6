import numpy as np
import torch


def _generate_sentence(model, w2i, decoding_strategy, init_word=" ", count_sentences=5, max_length=50, EOS="<eos>", device="cpu"):
    i2w = {id_: word for word, id_ in w2i.items()}
    model.eval()

    with torch.no_grad():
        for _ in range(count_sentences):
            input_word = torch.tensor([w2i[init_word]], dtype=torch.long).to(device)
            input_word = input_word.unsqueeze(0) 
            generated_words = [input_word.squeeze().item()]

            for _ in range(max_length):
                outputs, _ = model(input_word)
                outputs = outputs.squeeze(0)

                if decoding_strategy == "greedy":
                    predicted_word = torch.argmax(outputs, dim=-1)
                else:
                    probs = torch.nn.functional.softmax(outputs, dim=-1)
                    predicted_word = torch.multinomial(probs, 1).squeeze()
                    # probs = torch.nn.functional.softmax(outputs, dim=-1).cpu().numpy().flatten()
                    # predicted_word = np.random.choice(range(len(probs)), p=probs)

                if predicted_word.item() == w2i[EOS] or len(generated_words) == max_length:
                    generated_words.append(predicted_word.item())
                    break

                generated_words.append(predicted_word.item())
                input_word = predicted_word.unsqueeze(0)

            generated_sent = ' '.join([i2w[word_id] for word_id in generated_words])
            print(f"{decoding_strategy}: {generated_sent}")

def generate_sentence_greedy(*args, **kwargs):
    return _generate_sentence(*args, decoding_strategy="greedy", **kwargs)

def generate_sentence_sampling(*args, **kwargs):
    return _generate_sentence(*args, decoding_strategy="sampling", **kwargs)

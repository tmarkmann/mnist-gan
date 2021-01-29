import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imageio
import glob
import PIL

def plot_loss_acc(summary, output_name='performance', real_inception_score=0):
    # Plot loss
    fig, ((ax1, ax2), (ax3, _)) = plt.subplots(2, 2, sharex=True, figsize = (15, 10))

    ax1.set_title('Loss')
    ax1.set(xlabel='', ylabel='loss')
    ax1.plot(range(summary['epochs']), summary['gen_losses'], label='gen')
    ax1.plot(range(summary['epochs']), summary['disc_real_losses'], label='disc-real')
    ax1.plot(range(summary['epochs']), summary['disc_fake_losses'], label='disc-fake')
    ax1.legend()

    ax2.set_title('Discriminator Accuracy')
    ax2.set(xlabel='epoch', ylabel='accuracy')
    ax2.plot(range(summary['epochs']), summary['accuracies_real'], label='acc-real')
    ax2.plot(range(summary['epochs']), summary['accuracies_fake'], label='acc-fake')
    ax2.legend()
    
    ax3.set_title('Generator Inception Score')
    ax3.set(xlabel='epoch', ylabel='score')
    ax3.plot(range(summary['epochs']), summary['inception_score'], label='fake score')
    ax3.plot([0, summary['epochs']], [real_inception_score, real_inception_score], label='real score')
    ax3.legend()

    x_ax = ax1.axes.get_xaxis()  ## Get X axis
    x_ax.set_major_locator(ticker.MaxNLocator(integer=True))  ## Set major locators to integer values

    plt.tight_layout()
    plt.savefig(f'figures/{output_name}.png')
    plt.show()
    
def plot_loss_acc_fashion(summary, output_name='performance_fashion'):
    # Plot loss
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize = (10, 5))

    ax1.set_title('Loss')
    ax1.set(xlabel='', ylabel='loss')
    ax1.plot(range(summary['epochs']), summary['gen_losses'], label='gen')
    ax1.plot(range(summary['epochs']), summary['disc_real_losses'], label='disc-real')
    ax1.plot(range(summary['epochs']), summary['disc_fake_losses'], label='disc-fake')
    ax1.legend()

    ax2.set_title('Discriminator Accuracy')
    ax2.set(xlabel='epoch', ylabel='accuracy')
    ax2.plot(range(summary['epochs']), summary['accuracies_real'], label='acc-real')
    ax2.plot(range(summary['epochs']), summary['accuracies_fake'], label='acc-fake')
    ax2.legend()

    x_ax = ax1.axes.get_xaxis()  ## Get X axis
    x_ax.set_major_locator(ticker.MaxNLocator(integer=True))  ## Set major locators to integer values

    plt.tight_layout()
    plt.savefig(f'figures/{output_name}.png')
    plt.show()

def create_gif(image_path):
    # Use `imageio` to create an animated gif using the images saved during training.
    anim_file = '{}/dcgan.gif'.format(image_path)

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('{}/image*.png'.format(image_path))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

def display_image(epoch_no):
    # Display a single image using the epoch number
    return PIL.Image.open('mnist/image_at_epoch_{:04d}.png'.format(epoch_no))

def generate_and_save_images(model, epoch, test_input, output_dir):
    predictions = model(test_input, training=False) # Notice `training` is set to False. This is so all layers run in inference mode (batchnorm).
    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(output_dir, epoch))
    plt.show()

def plot_latent_space_steps(examples, n):
    # plot images
    fig, axs = plt.subplots(5, 10, figsize=(15, 10))
    for ax,im in zip(fig.axes, examples[0:50]):
        ax.imshow(im, cmap='gray')
        ax.grid(True)
        ax.axis('off')
    fig.suptitle('Exploring the latent space', fontsize=20)
    fig.text(0.5, 0.04, 'Steps in z space', ha='center', fontsize=15)
    fig.text(0.04, 0.5, 'Samples', va='center', rotation='vertical', fontsize=15)
    plt.show()
    
def get_empty_summary(epochs):
     return {
        "epochs": epochs,
        "gen_losses": list(),
        "disc_real_losses": list(),
        "disc_fake_losses": list(),
        "accuracies_real": list(),
        "accuracies_fake": list(),
        "inception_score": list(),
    }
    
def add_to_summary(summary, performance):
    summary["gen_losses"].append(performance["gen_loss"])
    summary["disc_real_losses"].append(performance["disc_real_loss"])
    summary["disc_fake_losses"].append(performance["disc_fake_loss"])
    summary["accuracies_real"].append(performance["acc_real"])
    summary["accuracies_fake"].append(performance["acc_fake"])
    summary["inception_score"].append(performance["inc_score"])
    return summary

    
def get_empty_performance():
    return {
        "gen_loss": 0,
        "disc_real_loss": 0,
        "disc_fake_loss": 0,
        "acc_real": 0,
        "acc_fake": 0,
        "inc_score": 0,
    }

def add_to_performance(perf, gen_loss, disc_real_loss, disc_fake_loss, acc_real, acc_fake, inc_score):
    perf["gen_loss"] += gen_loss
    perf["disc_real_loss"] += disc_real_loss
    perf["disc_fake_loss"] += disc_fake_loss
    perf["acc_real"] += acc_real
    perf["acc_fake"] += acc_fake
    perf["inc_score"] += inc_score
    return perf

def get_perf_mean(perf, n):
    for key, value in perf.items():
        perf[key] = value / n
    return perf
    
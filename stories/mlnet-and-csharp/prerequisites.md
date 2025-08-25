# Install The NET SDK

In this course we're going to use Microsoft NET. This is a modern cross-platform software development framework that runs on Windows, macOS, and Linux.

This means that you can build and run the labs in this course on either a Windows computer, a Mac, or a Linux machine. You do not need a virtual machine or an interface layer like Wine to run NET code.

We will set up, compile and run all applications directly from the command line. For this, we'll need to install the NET Software Development Kit (SDK).

{{< encrypt >}}

You can download the SDK here:
https://dotnet.microsoft.com/en-us/download/dotnet

![Download The NET SDK](../img/download-net.jpg)
{ .img-fluid .mb-4 }

Select the most recent standard- or long term support version, and then download the SDK installer for your operating system. Run the installer to install the SDK.

After the installer has finished you can check if everything got installed correctly by running the following command in your terminal or console app:

```bash
dotnet --info
```

You should see information about your operating system and the installed NET runtime and SDK versions.

Here's what I see when I run the command on my laptop (screenshot taken in April 2025):

![Test The NET SDK](../img/run-net.jpg)
{ .img-fluid .mb-4 }

You can see that I'm using a Windows laptop running Windows 10.0.26100 64-bit, and that I have the SDKs installed for NET version 7, 8 and 9. You can have multiple versions of the SDK installed side-by-side without them interfering with each other.

You should see similar output. You're now ready to start building NET applications. But let's install a nice integrated development environment next. 

{{< /encrypt >}}

# Install The Visual Studio Code Editor

We are going to use Visual Studio Code to build and edit our machine learning apps. Visual Studio Code is a very nice cross-platform integrated development environment based on Electron that runs on Windows, macOS, and Linux.

This means that you can edit, run, and debug all apps in this course using either a Windows computer, a Mac, or a Linux machine. There's no need to install anything special to work on the labs

{{< encrypt >}}

You can download the Visual Studio Code installer here:
https://code.visualstudio.com/

After installing Visual Studio Code and launching it, you'll see something like this:

![Run Visual Studio Code](../img/run-vscode.jpg)
{ .img-fluid .mb-4 }

And that's it! You now have the NET Software Development Kit and Visual Studio Code installed on your computer. With this, you're ready to start building machine learning applications in C#. 

{{< /encrypt >}}

# Set Up Copilot

In this lab, you will be collaborating with an AI agent that will generate C# code for you. You'll be using Github Copilot which is tightly integrated in Visual Studio Code. Once properly set up, Copilot can write and debug code for you, annotate your code with comments, build and test your apps, import and set up NuGet packages, and much more. 

To start, you'll need a Github account. 

{{< encrypt >}}

Head over to the [Github Signup](https://github.com/signup) page and set up a new account, if you do not already have one. 

Next, you'll need to set up a Github Copilot subscription. You have two choices:

- **GitHub Copilot Free**, which includes up to 2,000 code completions and 50 premium requests per month. 
- **GitHub Copilot Pro**, which includes unlimited code completions, access to premium models and up to 300 premium requests per month. You can sign up for a [30-day free trial of Pro](https://github.com/github-copilot/pro). 

The free subscription is perfectly fine for completing all of the assignments in this lab, but if you want, you can consider signing up for the 30-day trial of Copilot Pro to try it out. Another strategy is to subscribe to Free, and then [register your Anthropic or OpenAI API keys in Visual Studio Code](https://code.visualstudio.com/docs/copilot/language-models#_bring-your-own-language-model-key) to extend your access to premium models. That's what I am doing. 

Once you've made your choice, it's time to set up Copilot in Visual Studio Code.

Open the VS Code app and click on the Copilot status icon in the bottom right corner of the application. From the popup menu that appears, click 'Set up Copilot'. 

![Set up Copilot](../img/setup-copilot-status-bar.png)
{ .img-fluid .mb-4 }

Then click 'Sign in' to sign in to your GitHub account. If you are already signed in, a 'Use Copilot' button will appear and you should click that instead.

![Set up Copilot](../img/setup-copilot-sign-in.png)
{ .img-fluid .mb-4 }

If you haven't set up a Copilot subscription yet, you will automatically be signed up for the **Copilot Free** plan.

And that's it, you're done. You can now start using Copilot in Visual Studio Code. 

{{< /encrypt >}}

# Conclusion

This concludes the section on course prerequisites.

You now have the Microsoft NET SDK and the Visual Studio Code integrated development environment installed on your computer. You are ready to start developing your first machine learning application in C#.

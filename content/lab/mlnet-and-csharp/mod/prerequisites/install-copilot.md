---
title: "Set Up Copilot"
type: "lesson"
layout: "default"
sortkey: 30
---

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

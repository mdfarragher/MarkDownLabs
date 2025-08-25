---
title: "Install The NET SDK"
type: "lesson"
layout: "default"
sortkey: 10
---

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

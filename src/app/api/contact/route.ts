// app/api/contact/route.ts
import { NextRequest, NextResponse } from "next/server";
import nodemailer from "nodemailer";

function escapeHtml(input: string) {
  return input
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export async function POST(request: NextRequest) {
  try {
    const { name, email, subject, message } = await request.json();

    // Validate required fields
    if (!name || !email || !subject || !message) {
      return NextResponse.json({ error: "All fields are required" }, { status: 400 });
    }

    // Validate email format
    const emailRegex = /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i;
    if (!emailRegex.test(email)) {
      return NextResponse.json({ error: "Invalid email format" }, { status: 400 });
    }

    // Sanitize user content
    const safeName = escapeHtml(name);
    const safeEmail = escapeHtml(email);
    const safeSubject = escapeHtml(subject);
    const safeMessage = escapeHtml(message);

    // Transport (IONOS)
    const port = Number(process.env.SMTP_PORT || 587);
    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port,
      secure: port === 465, // SSL/TLS if 465
      auth: {
        user: process.env.SMTP_USER, // contact@heliotheanalyst.co.uk
        pass: process.env.SMTP_PASS!,
      },
    });

    // 1) Notify YOU (to contact inbox) — sent from noreply@
    const adminMailOptions = {
      from: process.env.CONTACT_EMAIL_FROM, // noreply@
      to: process.env.CONTACT_EMAIL_TO,     // contact@
      subject: `Contact Form: ${subject}`,
      text: `New contact form submission

Name: ${name}
Email: ${email}
Subject: ${subject}

Message:
${message}

(Reply to this email will go to ${email})
`,
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #1A0E24; border-bottom: 2px solid #58E6E6; padding-bottom: 10px;">
            New Contact Form Submission
          </h2>
          <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
            <p><strong>Name:</strong> ${safeName}</p>
            <p><strong>Email:</strong> ${safeEmail}</p>
            <p><strong>Subject:</strong> ${safeSubject}</p>
          </div>
          <div style="margin: 20px 0;">
            <h3 style="color: #1A0E24;">Message:</h3>
            <div style="background-color: #ffffff; padding: 15px; border-left: 4px solid #58E6E6; border-radius: 4px;">
              ${safeMessage.replace(/\n/g, "<br>")}
            </div>
          </div>
          <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef; color: #6c757d; font-size: 12px;">
            <p>This email was sent from your portfolio contact form.</p>
            <p>Reply directly to this email to respond to ${safeName} at ${safeEmail}</p>
          </div>
        </div>
      `,
      replyTo: email,
    };

    // 2) Auto-reply to the sender — sent from contact@
    const userAutoReplyOptions = {
      from: process.env.CONTACT_EMAIL_TO, // contact@
      to: email,
      subject: `Thanks for reaching out — ${process.env.SITE_NAME || "HelioTheAnalyst"}`,
      text: `Hi ${name},

Thanks for getting in touch! I’ve received your message and will get back to you soon.

Summary of your submission:
Subject: ${subject}
Message:
${message}

If you need to add anything else, just reply to this email.

— Helio
www.heliotheanalyst.co.uk
`,
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #1A0E24;">Thanks for reaching out!</h2>
          <p>Hi ${safeName},</p>
          <p>Thanks for getting in touch! I’ve received your message and will get back to you soon.</p>
          <div style="background:#f8f9fa; padding: 16px; border-radius: 8px; margin: 16px 0;">
            <p style="margin:0;"><strong>Subject:</strong> ${safeSubject}</p>
            <p style="margin:12px 0 0 0;"><strong>Your message:</strong></p>
            <div style="background:#fff; padding:12px; border-left:4px solid #58E6E6; border-radius:4px;">
              ${safeMessage.replace(/\n/g, "<br>")}
            </div>
          </div>
          <p>If you need to add anything else, just reply to this email.</p>
          <p style="margin-top:24px;">— Helio<br/>
          <a href="https://www.heliotheanalyst.co.uk">www.heliotheanalyst.co.uk</a></p>
          <hr style="border:none; border-top:1px solid #e9ecef; margin:20px 0;">
          <p style="color:#6c757d; font-size:12px;">This confirmation was sent from contact@heliotheanalyst.co.uk.</p>
        </div>
      `,
      replyTo: process.env.CONTACT_EMAIL_TO, // replies go to your contact inbox
      headers: {
        "List-Unsubscribe": `<mailto:${process.env.CONTACT_EMAIL_TO}>`,
      },
    };

    await Promise.all([
      transporter.sendMail(adminMailOptions),
      transporter.sendMail(userAutoReplyOptions),
    ]);

    return NextResponse.json({ message: "Emails sent successfully" }, { status: 200 });
  } catch (error) {
    console.error("Error sending email:", error);
    return NextResponse.json({ error: "Failed to send email" }, { status: 500 });
  }
}

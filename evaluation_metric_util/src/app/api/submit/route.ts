import { NextRequest, NextResponse } from 'next/server';
import { google } from 'googleapis';

type SheetForm = {
  no: number;
  filename: string;
  directory: string;
  rating: number;
};

// Named export for POST method
export async function POST(req: NextRequest) {
  const body: SheetForm[] = await req.json(); // Extract body from the request

  try {
    // Prepare Google Sheets authentication
    const auth = new google.auth.GoogleAuth({
      credentials: {
        client_email: process.env.GOOGLE_CLIENT_EMAIL,
        private_key: process.env.GOOGLE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
      },
      scopes: [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file',
      ],
    });

    const sheets = google.sheets({ version: 'v4', auth });

    // Append values to the spreadsheet
    const response = await sheets.spreadsheets.values.append({
      spreadsheetId: process.env.GOOGLE_SHEET_ID,
      range: 'B:D', // Adjusted range for filename, directory, and rating
      valueInputOption: 'USER_ENTERED',
      requestBody: {
        values: body.map((item) => [item.filename, item.directory, item.rating]),
      },
    });

    return NextResponse.json({ success: true, response: response.data });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}

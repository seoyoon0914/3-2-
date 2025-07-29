# MyHomepage - Modern Next.js Website

A modern, responsive website built with Next.js, React, and TypeScript. Deployed on Vercel with serverless architecture.

## 🚀 Features

- **Next.js 15** with App Router
- **React 19** with TypeScript
- **Tailwind CSS** for modern styling
- **Vercel Serverless** deployment
- **Responsive Design** for all devices
- **SEO Optimized** with metadata
- **Fast Performance** with optimized builds

## 🛠️ Tech Stack

- **Framework**: Next.js 15
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Deployment**: Vercel
- **Architecture**: Serverless

## 📦 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd homepage
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🚀 Deployment

This project is configured for immediate deployment on Vercel:

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Deploy automatically

### Manual Deployment

```bash
# Build the project
npm run build

# Deploy to Vercel
vercel --prod
```

## 📁 Project Structure

```
homepage/
├── src/
│   ├── app/
│   │   ├── api/           # Serverless API routes
│   │   ├── globals.css    # Global styles
│   │   ├── layout.tsx     # Root layout
│   │   └── page.tsx       # Home page
│   └── ...
├── public/                # Static assets
├── vercel.json           # Vercel configuration
├── next.config.ts        # Next.js configuration
└── package.json
```

## 🔧 Configuration

### Vercel Configuration (`vercel.json`)
- Serverless function configuration
- Route handling
- Runtime settings

### Next.js Configuration (`next.config.ts`)
- Standalone output for Vercel
- Image optimization settings
- Security headers

## 📱 API Endpoints

- `GET /api/hello` - Returns serverless function status
- `POST /api/hello` - Accepts JSON data and returns confirmation

## 🎨 Customization

### Styling
- Modify `src/app/globals.css` for global styles
- Use Tailwind CSS classes for component styling
- Update color scheme in the main page component

### Content
- Edit `src/app/page.tsx` to modify the homepage content
- Update metadata in `src/app/layout.tsx`
- Add new pages in the `src/app` directory

### API
- Add new API routes in `src/app/api/`
- Each route should export GET, POST, PUT, DELETE handlers

## 🔍 SEO & Performance

- Optimized metadata for search engines
- Open Graph tags for social media
- Twitter Card support
- Fast loading with Next.js optimizations
- Serverless architecture for scalability

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For support, please open an issue in the GitHub repository.

---

Built with ❤️ using Next.js and deployed on Vercel

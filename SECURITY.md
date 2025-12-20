# Security Considerations

This Jekyll site implements several security best practices:

## Security Headers

The `_includes/head.html` file includes:

- **Content-Security-Policy (CSP)**: Restricts resource loading to prevent XSS attacks
- **X-Content-Type-Options**: Prevents MIME type sniffing
- **X-Frame-Options**: Protects against clickjacking
- **X-XSS-Protection**: Enables browser XSS filtering
- **Referrer Policy**: Controls referrer information

## Jekyll Configuration

In `_config.yml`:

- **strict_front_matter**: Enforces valid YAML front matter
- **safe**: Disables custom plugins (required for GitHub Pages)

## Best Practices

- Keep dependencies updated via Dependabot
- Use HTTPS for all external resources
- Sanitize user inputs if adding forms
- Regular security audits of dependencies

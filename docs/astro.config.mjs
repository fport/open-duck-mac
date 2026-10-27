// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://fport.github.io",
  base: "/open-duck-mac",
  integrations: [
    starlight({
      title: "open-duck-mac",
      description: "Open Duck Mini v2'ye bir Mac'te, sıfırdan, yalnız işlemciyle yürümeyi öğretmek.",
      defaultLocale: "root",
      locales: {
        root: { label: "Türkçe", lang: "tr" },
        en: { label: "English", lang: "en" },
      },
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/fport/open-duck-mac" },
        { icon: "youtube", label: "YouTube", href: "https://www.youtube.com/@getporti" },
      ],
      editLink: { baseUrl: "https://github.com/fport/open-duck-mac/edit/main/docs/" },
      lastUpdated: true,
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Başla",
          translations: { en: "Start" },
          items: [
            { label: "Ana sayfa", translations: { en: "Home" }, link: "/" },
            { label: "Kurulum", translations: { en: "Setup" }, slug: "setup" },
            { label: "Kavramlar", translations: { en: "Concepts" }, slug: "concepts" },
          ],
        },
        {
          label: "Deneyler",
          translations: { en: "Experiments" },
          items: [
            { label: "1. Yürümeyi öğrenmek", translations: { en: "1. Learning to walk" }, slug: "part-1-walking" },
            { label: "2. Reward hacking", translations: { en: "2. Reward hacking" }, slug: "part-2-reward-hacking" },
            { label: "3. Kutudaki motor gücü", translations: { en: "3. The motor on the box" }, slug: "part-3-sim2real" },
            { label: "Sen ne yapabilirsin?", translations: { en: "What can you try?" }, slug: "experiments" },
          ],
        },
        {
          label: "Ekosistem",
          translations: { en: "Ecosystem" },
          items: [
            { label: "Microduck", translations: { en: "Microduck" }, slug: "microduck" },
          ],
        },
        {
          label: "Başvuru",
          translations: { en: "Reference" },
          items: [
            { label: "Sorunlar ve SSS", translations: { en: "Troubleshooting & FAQ" }, slug: "faq" },
            { label: "Betikler ve dosyalar", translations: { en: "Scripts and files" }, slug: "reference" },
          ],
        },
      ],
    }),
  ],
});

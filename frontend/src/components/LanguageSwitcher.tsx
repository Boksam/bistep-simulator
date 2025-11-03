import { useTranslation } from "react-i18next";
import "./LanguageSwitcher.css";

const LanguageSwitcher = () => {
  const { i18n } = useTranslation();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
    localStorage.setItem("language", lng);
  };

  return (
    <div className="language-switcher">
      <button
        className={i18n.language === "ko" ? "active" : ""}
        onClick={() => changeLanguage("ko")}
      >
        한국어
      </button>
      <button
        className={i18n.language === "en" ? "active" : ""}
        onClick={() => changeLanguage("en")}
      >
        English
      </button>
    </div>
  );
};

export default LanguageSwitcher;

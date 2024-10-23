package org.michael;

import org.projog.api.Projog;
import org.projog.api.QueryResult;

import java.io.File;
import java.util.Objects;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Projog projog = new Projog();

        // Загружаем базу знаний из файла
        try {
            projog.consultFile(
                    new File(Objects.requireNonNull(Main.class.getClassLoader().getResource("base.pl")).getFile())
            );
        } catch (Exception e) {
            System.out.println("Ошибка при загрузке базы знаний: " + e.getMessage());
            return;
        }

        // Начинаем диалог с пользователем
        Scanner scanner = new Scanner(System.in);
        System.out.println("Добро пожаловать в нашу систему рекомендаций!");
        System.out.println("Пожалуйста, ответьте на несколько вопросов:");

        boolean running = true;

        while (running) {
            // Выводим меню опций
            System.out.println();
            System.out.println("Выберите опцию:");
            System.out.println("1) Персонажи с мощными киберимплантами");
            System.out.println("2) Персонажи, являющиеся врагами корпорации Арасака");
            System.out.println("3) Персонажи, работающие на корпорации");
            System.out.println("4) Персонажи, являющиеся членами банд");
            System.out.println("5) Друзья V");
            System.out.println("6) Персонажи, находящиеся в Найт-Сити");
            System.out.println("7) Персонажи, вооруженные опасным оружием");
            System.out.println("8) Персонажи, у которых есть киберимпланты");
            System.out.println("9) Персонажи, связанные с заданным персонажем");
            System.out.println("0) Выход");
            System.out.print("Введите номер опции (0-9): ");

            String choice = scanner.nextLine().trim();

            String query = null;
            String description = null;

            switch (choice) {
                case "1":
                    // Запрос персонажей с мощными киберимплантами
                    query = "has_powerful_cyberware(Character).";
                    description = "Персонажи с мощными киберимплантами:";
                    break;
                case "2":
                    // Запрос персонажей, являющихся врагами корпорации Арасака
                    query = "is_enemy_of_arasaka(Character).";
                    description = "Персонажи, являющиеся врагами корпорации Арасака:";
                    break;
                case "3":
                    // Запрос персонажей, работающих на корпорации
                    query = "is_corporate_agent(Character).";
                    description = "Персонажи, работающие на корпорации:";
                    break;
                case "4":
                    // Запрос персонажей, являющихся членами банд
                    query = "is_gang_member(Character).";
                    description = "Персонажи, являющиеся членами банд:";
                    break;
                case "5":
                    // Запрос друзей V
                    query = "is_friend_of_v(Character).";
                    description = "Друзья V:";
                    break;
                case "6":
                    // Запрос персонажей, находящихся в Найт-Сити
                    query = "is_in_night_city(Character).";
                    description = "Персонажи, находящиеся в Найт-Сити:";
                    break;
                case "7":
                    // Запрос персонажей, вооруженных опасным оружием
                    query = "is_heavily_armed(Character).";
                    description = "Персонажи, вооруженные опасным оружием:";
                    break;
                case "8":
                    // Запрос персонажей, у которых есть киберимпланты
                    query = "has_cyberware(Character).";
                    description = "Персонажи с киберимплантами:";
                    break;
                case "9":
                    // Запрос персонажей, связанных с заданным персонажем
                    System.out.print("Введите имя персонажа: ");
                    String inputCharacter = scanner.nextLine().trim();
                    query = "has_relationship('" + inputCharacter + "', Character).";
                    description = "Персонажи, связанные с " + inputCharacter + ":";
                    break;
                case "0":
                    running = false;
                    continue;
                default:
                    System.out.println("Некорректный выбор.");
                    continue;
            }
            // Выполняем запрос с помощью Projog
            try {
                System.out.println();
                System.out.println(description);
                QueryResult result = projog.executeQuery(query);

                boolean found = false;

                while (result.next()) {
                    // Получаем значение переменной Character
                    String character = result.getAtomName("Character");
                    System.out.println("- " + character);
                    found = true;
                }

                if (!found) {
                    System.out.println("Ни одного персонажа не найдено.");
                }

            } catch (Exception e) {
                System.out.println("Ошибка при выполнении запроса: " + e.getMessage());
            }
        }
        scanner.close();
    }
}
